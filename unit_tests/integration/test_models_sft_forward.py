"""
Integration test: synthetic Parquet -> PairedFeed -> SFT forward pass.

Uses TinyModel (pure PyTorch, no HF download) and a lightweight mock
tokenizer so the test runs fully offline.
"""
import os
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
import pytest
from torch.utils.data import DataLoader

from unit_tests.models import TinyModel


VOCAB_SIZE = 64


class _MockTokenizer:
    """Minimal tokenizer that satisfies PairedFeed's interface."""

    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def apply_chat_template(
        self, conversation, add_generation_prompt=False, tokenize=True, return_tensors=None
    ):
        text = " ".join(t["content"] for t in conversation)
        if add_generation_prompt:
            text += " [A]"
        ids = self._encode(text)
        if tokenize and return_tensors == "pt":
            return torch.tensor([ids])
        return text

    def __call__(self, text, return_tensors=None, add_special_tokens=True):
        ids = self._encode(text)
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def _encode(self, text: str):
        # Deterministic word-level encoding, avoids pad (0) and eos (1) ids.
        ids = [(abs(hash(w)) % (VOCAB_SIZE - 2)) + 2 for w in text.split()]
        return ids or [2]


def test_synthetic_data_paired_feed_and_forward_pass():
    from data_prep.synthetic import build_synthetic_dataframe
    PairedFeed = pytest.importorskip(
        "data_feeds.paired",
        reason="Requires 'datasets'/'huggingface_hub' runtime deps.",
    ).PairedFeed

    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Write synthetic Parquet
        parquet_path = os.path.join(tmpdir, "smoke.parquet")
        df = build_synthetic_dataframe(n=8)
        df.to_parquet(parquet_path, index=False)

        # 2. Build PairedFeed
        tokenizer = _MockTokenizer()
        dataset = PairedFeed(
            prompt_key="prompt",
            answer_key="answer",
            max_seq_len=64,
            tokenizer=tokenizer,
            data_path=parquet_path,
        )
        assert len(dataset) == 8

        dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
        batch = next(iter(dataloader))

        assert "input_ids" in batch
        assert "attn_mask" in batch
        assert "loss_mask" in batch
        assert batch["input_ids"].shape == (4, 64)
        assert batch["loss_mask"].shape == (4, 63)

        # 3. Forward pass through TinyModel
        model = TinyModel(vocab_size=VOCAB_SIZE, hidden_dim=16)
        model.eval()

        input_ids = batch["input_ids"].clamp(0, VOCAB_SIZE - 1)
        attn_mask = batch["attn_mask"]

        with torch.no_grad():
            output = model(input_ids, attention_mask=attn_mask)

        assert output.logits.shape == (4, 64, VOCAB_SIZE)
        assert torch.isfinite(output.logits).all()


def test_sft_train_step_with_synthetic_data():
    """One full SFT train step: forward + loss + backward + optimizer step."""
    from data_prep.synthetic import build_synthetic_dataframe
    PairedFeed = pytest.importorskip(
        "data_feeds.paired",
        reason="Requires 'datasets'/'huggingface_hub' runtime deps.",
    ).PairedFeed
    from algs.SFT.sft import SFT

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "smoke.parquet")
        df = build_synthetic_dataframe(n=4)
        df.to_parquet(parquet_path, index=False)

        tokenizer = _MockTokenizer()
        dataset = PairedFeed(
            prompt_key="prompt",
            answer_key="answer",
            max_seq_len=64,
            tokenizer=tokenizer,
            data_path=parquet_path,
        )
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(dataloader))

        model = TinyModel(vocab_size=VOCAB_SIZE, hidden_dim=16)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        original_weight = model.lm_head.weight.detach().clone()

        # Wrap model in a lightweight shim that mimics DeepSpeed's engine interface.
        class _EngineShim:
            def __init__(self, m, opt):
                self.module = m
                self._opt = opt
                self.device = torch.device("cpu")

            def __call__(self, *a, **kw):
                return self.module(*a, **kw)

            def backward(self, loss):
                loss.backward()

            def step(self):
                self._opt.step()
                self._opt.zero_grad()

        from models.adapters.text_causal_lm import TextCausalLMAdapter
        engine = _EngineShim(model, optimizer)
        alg = SFT(model_engine=engine, optimizer=optimizer, model_adapter=TextCausalLMAdapter(), normalize_loss=True)

        micro_batch = {k: v.clamp(0, VOCAB_SIZE - 1) if k == "input_ids" else v for k, v in batch.items()}
        ga_denom = float(micro_batch["loss_mask"].sum().item())
        metrics = alg.train_step(micro_batch, ga_denom=ga_denom, ga_steps=1)

        assert torch.isfinite(torch.tensor(metrics["loss"]))
        # Parameters should have changed after the update
        assert not torch.equal(model.lm_head.weight, original_weight)
