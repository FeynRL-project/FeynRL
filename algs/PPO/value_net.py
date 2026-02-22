import torch
import torch.nn as nn
from types import SimpleNamespace

class ValueNetwork(nn.Module):
    '''
        Wraps a LM backbone with a scalar value head for value function.
        Replaces the LM head (hidden_dim -> vocab_size) with a value head (hidden_dim -> 1).
        Output: logits of shape [B, T, 1] so that value_forward's .squeeze(-1) yields [B, T].
    '''
    def __init__(self, base_model):
        super().__init__()
        self.config = base_model.config

        # Extract the transformer backbone without LM head.
        # Most HF models expose the backbone as .model (LLaMA, Gemma, Mistral, Qwen)
        # or .transformer (GPT-2, GPT-Neo).
        if hasattr(base_model, 'model'):
            self.backbone = base_model.model

        elif hasattr(base_model, 'transformer'):
            self.backbone = base_model.transformer

        else:
            raise ValueError(
                f"Cannot find backbone in {type(base_model).__name__}. "
                "Expected .model or .transformer attribute."
            )

        # Add a value head with hidden_dim equals to 1.
        self.value_head = nn.Linear(base_model.config.hidden_size, 1, bias=False)
        # Initialize near-zero so initial values don't dominate early training
        nn.init.zeros_(self.value_head.weight)

        # Keep head dtype/device aligned with backbone to avoid dtype errors.
        first_param = next(self.backbone.parameters(), None)
        if first_param is not None:
            self.value_head.to(device=first_param.device, dtype=first_param.dtype)

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=False):
        '''
            input_ids, attention_mask, position_ids: [B, T]
            return: SimpleNamespace(logits=[B, T, 1])
        '''
        # [B, T, hidden_dim]
        outputs = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                use_cache=use_cache,
                                )
        # [B, T, hidden_dim]
        hidden_states = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]

        # Since mixed-precision/autocast can still change activation dtype at runtime,
        # we do guard against mixed-precision mismatches.
        if hidden_states.dtype != self.value_head.weight.dtype:
            hidden_states = hidden_states.to(self.value_head.weight.dtype)

        # [B, T, hidden_dim] -->[B, T, 1]
        values = self.value_head(hidden_states) 

        # HF models return objects like outputs that have a .logits attribute. 
        # Since ValueModel uses a custom head, we need something that also has .logits 
        # so value_forward can do output.logits without caring whether it's a policy model or value model
        # and to be consistent with the policy model's forward pass.
        return SimpleNamespace(logits=values)

    def gradient_checkpointing_enable(self):
        '''
            Enable gradient checkpointing for the backbone.
            ValueNetwork is nn.Module, not a HF model, so it doesn't have
            gradient_checkpointing_enable() method. We need to call it on the backbone.
        '''
        if hasattr(self.backbone, 'gradient_checkpointing_enable'):
            self.backbone.gradient_checkpointing_enable()

    def enable_input_require_grads(self):
        '''
            Delegate to backbone so common.py _load_single_model can call this
            on ValueNetwork the same way it does on HF models.
        '''
        if hasattr(self.backbone, 'enable_input_require_grads'):
            self.backbone.enable_input_require_grads()

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained('google/gemma-3-1b-it')
    base_model  = AutoModelForCausalLM.from_pretrained(
                                        'google/gemma-3-1b-it',
                                        dtype=torch.bfloat16,
                                        trust_remote_code=True,
                                        config=config,
                                        )
    print("The original network with LM head")
    print(base_model)
    value_model = ValueNetwork(base_model=base_model)
    print(50*'=')
    print("Here is the value network without LM head")
    print(value_model)

    # Test forward pass
    B = 3
    T = 7
    input_ids = torch.randint(0, 1000, (B, T)).to(torch.long)
    attention_mask = torch.ones(B, T).to(torch.long)
    outputs = value_model(input_ids, attention_mask)
    print('The shape of the logits is: ', outputs.logits.shape)
    
