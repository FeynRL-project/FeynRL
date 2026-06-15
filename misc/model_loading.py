from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModelForImageTextToText, AutoProcessor
from misc.utils import safe_string_to_torch_dtype

def resolve_dtype(model_dtype):
    '''
        Accept either a string (e.g. "bfloat16", as used by main_sl/main_cl) or an
        already-resolved torch.dtype (as used by the RL alg) and return a torch.dtype.
    '''
    assert model_dtype != 'auto', "dtype must not be auto to avoid any precision issues"
    if isinstance(model_dtype, str):
        return safe_string_to_torch_dtype(model_dtype)
    # already a torch.dtype
    return model_dtype

def build_hf_model(model_path, model_dtype, model_class, trust_remote_code, attn_impl):
    '''
        Load the bare HF model for a given model_class.
        This is the single place where the text-only vs multi-modal model class is selected, 
        so SFT, CL, and RL all get the same behavior. It deliberately does NOT apply PEFT, 
        gradient checkpointing, or DeepSpeed wrapping adn it must be handled by callers.
        Args:
            model_path:        HF hub id or local path.
            model_dtype:       bfloat16/float32/... string or a torch.dtype.
            model_class:       llm, vlm, or whatever validated in load.py.
            trust_remote_code: forwarded to AutoConfig/AutoModel.
            attn_impl:         None, '', 'eager', or 'flash_attention_2'. For vlm it maps
                               to impls, e.g. {"text_config": "flash_attention_2", "vision_config": "eager"}.

        Returns:
            A torch.nn.Module (the HF model), uninitialized w.r.t. PEFT/DeepSpeed.
    '''
    dtype = resolve_dtype(model_dtype)
    # attn_impl is a string for llm and for vlm it is dict.
    attn  = None if attn_impl == '' else attn_impl

    # Load config first so custom architectures resolve consistently across all paradigms.
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)

    if model_class == "vlm":
        model_cls = AutoModelForImageTextToText

    elif model_class == "llm":
        model_cls = AutoModelForCausalLM

    else:
        raise ValueError(f"Unsupported model_class '{model_class}' (expected llm or vlm for now)")

    model = model_cls.from_pretrained(model_path,
                                      dtype=dtype,
                                      trust_remote_code=trust_remote_code,
                                      config=config,
                                      attn_implementation=attn)
    return model


def load_tokenizer_or_processor(model_path, model_class="llm", trust_remote_code=False):
    '''
        Load the text tokenizer for a model_class, plus the multi-modal processor
        when applicable.

        For vlm, the AutoProcessor bundles the tokenizer (processor.tokenizer) and
        the image/audio pre-processing; we return both so callers can keep using the
        tokenizer for pad/eos logic while passing the processor to the data feed.

        Returns:
            (tokenizer, processor) where processor is None for llm.
    '''
    if model_class == "vlm":
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return processor.tokenizer, processor

    elif model_class == "llm":
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return tokenizer, None

    else:
        raise ValueError(f"Unsupported model_class '{model_class}' (expected llm or vlm for now)")


def ensure_pad_token(tokenizer, *models, rank=0):
    '''
        Make sure the tokenizer has a pad token and sync pad_token_id into each
        model's config so exported checkpoints are consistent, since vllm and similar
        read pad_token_id from config.json, not the tokenizer.
    '''
    if tokenizer.pad_token_id is None:
        if rank == 0:
            print("Warning: Pad token is not present, using eos token as pad token")

        if getattr(tokenizer, 'eos_token', None) is not None:
            # prefer explicit token if available
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        else:
            # fallback to eos token id
            tokenizer.pad_token_id = tokenizer.eos_token_id

    if tokenizer.pad_token_id is None:
        return

    # For multi-modal (vlm) models the LM fields (incl. pad_token_id) live under
    # config.text_config, while some consumers read the top-level config. We set
    # BOTH the top-level config and text_config (when present) so every reader
    # agrees. text-only (llm) models have no text_config, so this is identical to
    # the previous inline logic that set the top-level config.
    for model in models:
        if model is None:
            continue
        cfg = model.config
        # Set on the top-level config AND the nested text_config (vlm) so both
        # top-level config.json readers and LM-subconfig readers see the pad id.
        targets = [cfg]
        sub = getattr(cfg, 'text_config', None)
        if sub is not None:
            targets.append(sub)
        for target in targets:
            if getattr(target, 'pad_token_id', None) is None:
                target.pad_token_id = tokenizer.pad_token_id


def smoke_test(model_path, model_class, attn_impl=None, model_dtype="bfloat16"):
    '''
        Load a model + tokenizer/processor, run the pad-token sync, and print a concise
        summary with a few sanity checks. Used by the manual smoke test below.
    '''
    print(f"\n=== {model_class.upper()}: {model_path} ===")
    model = build_hf_model(model_path=model_path,
                           model_dtype=model_dtype,
                           model_class=model_class,
                           trust_remote_code=False,
                           attn_impl=attn_impl)
    tokenizer, processor = load_tokenizer_or_processor(model_path=model_path,
                                                       model_class=model_class,
                                                       trust_remote_code=False)
    ensure_pad_token(tokenizer, model, rank=0)

    # sanity checks
    assert tokenizer.pad_token_id is not None, "pad token was not set"
    if model_class == "vlm":
        assert processor is not None, "vlm must return a processor"
    else:
        assert processor is None, "llm must not return a processor"

    n_params = sum(p.numel() for p in model.parameters())
    text_cfg = getattr(model.config, "text_config", None)
    print(f"  model class     : {model.__class__.__name__}")
    print(f"  param dtype      : {next(model.parameters()).dtype}")
    print(f"  num params       : {n_params / 1e9:.2f}B")
    print(f"  processor        : {processor.__class__.__name__ if processor is not None else None}")
    print(f"  tokenizer pad id : {tokenizer.pad_token_id}")
    print(f"  config pad id    : {model.config.pad_token_id}")
    if text_cfg is not None:
        print(f"  text_config pad  : {text_cfg.pad_token_id}")


if __name__ == "__main__":
    # Manual smoke test for the llm and vlm load paths.
    # To Run: python -m misc.model_loading
    smoke_test("google/gemma-3-1b-it", model_class="llm")
    smoke_test("Qwen/Qwen2-VL-2B-Instruct", model_class="vlm")
    smoke_test("HuggingFaceTB/SmolVLM-256M-Instruct", model_class="vlm")
    print('Done.')

