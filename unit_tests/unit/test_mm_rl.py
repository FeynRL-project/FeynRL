from __future__ import annotations
import io
from unittest.mock import MagicMock
import pandas as pd
from PIL import Image


class _DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False, **kwargs):
        text = "|".join(f"{t.get('role')}:{t.get('content')}" for t in conversation)
        if tokenize:
            return [1] * min(32, max(1, len(text) // 3))
        return text


def _png_bytes() -> bytes:
    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_image_prompts_feed_returns_mm_dict(tmp_path):
    from data_feeds.image_prompts import ImagePromptsFeed

    df = pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "what is in the image?"}],
                "solution": "red square",
                "image_bytes": _png_bytes(),
            }
        ]
    )
    p = tmp_path / "mm.parquet"
    df.to_parquet(p, index=False)

    ds = ImagePromptsFeed(
        prompt_key="prompt",
        tokenizer=_DummyTokenizer(),
        max_seq_len=256,
        data_path=str(p),
        solution_key="solution",
    )
    item = ds[0]
    assert "prompt" in item
    assert "multi_modal_data" in item
    assert "prompt_token_ids" not in item
    assert "solution" in item


def test_image_prompts_feed_calls_prepare_messages(tmp_path):
    from data_feeds.image_prompts import ImagePromptsFeed

    adapter = MagicMock()
    adapter.prepare_messages.side_effect = lambda msg: [
        {"role": "user", "content": "<|vision_start|><|image_pad|><|vision_end|>" + msg[0]["content"]}
    ]

    df = pd.DataFrame([{"prompt": [{"role": "user", "content": "describe"}], "image_bytes": _png_bytes()}])
    p = tmp_path / "mm.parquet"
    df.to_parquet(p, index=False)

    ds = ImagePromptsFeed(
        prompt_key="prompt",
        tokenizer=_DummyTokenizer(),
        max_seq_len=256,
        data_path=str(p),
        adapter=adapter,
    )
    item = ds[0]
    adapter.prepare_messages.assert_called_once()
    assert "<|image_pad|>" in item["prompt"]


def test_prompt_sampler_dataset_kwargs_passthrough():
    from data_feeds.mixed_sampler import create_prompt_dataset_and_sampler

    class DummyDS:
        def __init__(self, prompt_key, tokenizer, max_seq_len, data_path, solution_key=None, foo=None):
            self.foo = foo
            self.prompt_key = prompt_key
            self.solution_key = solution_key
            self.max_seq_len = max_seq_len
            self.tokenizer = tokenizer
            self.data_path = data_path

        def __len__(self):
            return 1

        def collate_fn(self, batch):
            return batch

    ds, sampler, collate_fn = create_prompt_dataset_and_sampler(
        data_paths=["/tmp/does_not_need_to_exist.parquet"],
        prompt_key="prompt",
        solution_key="solution",
        max_seq_len=128,
        tokenizer=_DummyTokenizer(),
        train_ratios={"does_not_need_to_exist": 1.0},
        seed=0,
        local_batch_size=1,
        dataset_cls=DummyDS,
        dynamic_ratio_every_step=False,
        steps_per_epoch=1,
        dataset_kwargs={"foo": "bar"},
    )
    assert getattr(ds.datasets[0], "foo") == "bar"
    assert callable(collate_fn)
    assert len(list(iter(sampler))) == 1


def test_common_load_single_model_uses_registry_loader(monkeypatch):
    from algs.RL.common import COMMON
    import torch
    from types import SimpleNamespace

    c = COMMON()
    c.attn_impl = None
    c.trust_remote_code = True
    c.peft_config = SimpleNamespace(use_peft=False, peft_type=None)
    c.gradient_checkpointing = False
    c.alg_name = "TEST"
    c.model_class = "qwen2_5_vl"

    loader = MagicMock(return_value=(MagicMock(), MagicMock(), MagicMock()))
    get_loader = MagicMock(return_value=loader)
    monkeypatch.setattr("models.registry.get_loader", get_loader)

    _ = c.load_single_model(model_path="dummy", dtype=torch.bfloat16, model_name="policy")
    get_loader.assert_called_with("qwen2_5_vl")
