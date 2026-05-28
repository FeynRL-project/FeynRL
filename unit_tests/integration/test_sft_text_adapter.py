"""
Integration smoke test: PairedFeed -> TextCausalLMAdapter -> SFT train step.

Runs fully offline using TinyModel and a minimal mock tokenizer.
"""

import os
import tempfile

import pytest
import torch
import torch.optim as optim

from algs.SFT.sft import SFT
from models.adapters.text_causal_lm import TextCausalLMAdapter
from unit_tests.models import TinyModel


VOCAB_SIZE = 64


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "<eos>"

    def apply_chat_template(self, conversation, add_generation_prompt=False, tokenize=True, return_tensors=None):
        text = " ".join(t["content"] for t in conversation)
        if add_generation_prompt:
            text += " [A]"
        ids = self._encode(text)
        if tokenize and return_tensors == "pt":
            return torch.tensor([ids])
        return text

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        ids = self._encode(text)
        t = torch.tensor([ids])
        return {"input_ids": t, "attention_mask": torch.ones_like(t)}

    def _encode(self, text: str):
        ids = [(abs(hash(w)) % (VOCAB_SIZE - 2)) + 2 for w in text.split()]
        return ids or [2]


def test_sft_train_step_uses_adapter_forward():
    PairedFeed = pytest.importorskip(
        "data_feeds.paired",
        reason="Requires 'datasets'/'huggingface_hub' runtime deps for PairedFeed.",
    ).PairedFeed

    with tempfile.TemporaryDirectory() as tmpdir:
        pd = pytest.importorskip("pandas", reason="pandas required to write parquet fixture")
        parquet_path = os.path.join(tmpdir, "smoke.parquet")
        df = pd.DataFrame(
            [
                {"prompt": [{"role": "user", "content": "hi"}], "answer": "hello"},
                {"prompt": [{"role": "user", "content": "math"}], "answer": "2"},
            ]
        )
        df.to_parquet(parquet_path, index=False)

        tokenizer = _MockTokenizer()
        dataset = PairedFeed(
            prompt_key="prompt",
            answer_key="answer",
            max_seq_len=32,
            tokenizer=tokenizer,
            data_path=parquet_path,
        )
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)))

        model = TinyModel(vocab_size=VOCAB_SIZE, hidden_dim=16)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        class _EngineShim:
            def __init__(self, module, opt):
                self.module = module
                self._opt = opt
                self.device = torch.device("cpu")

            def __call__(self, *a, **kw):
                return self.module(*a, **kw)

            def backward(self, loss):
                loss.backward()

            def step(self):
                self._opt.step()
                self._opt.zero_grad()

        engine = _EngineShim(model, optimizer)
        adapter = TextCausalLMAdapter()
        alg = SFT(model_engine=engine, optimizer=optimizer, normalize_loss=True, model_adapter=adapter)

        # Clamp ids into vocab.
        micro_batch = {k: (v.clamp(0, VOCAB_SIZE - 1) if k == "input_ids" else v) for k, v in batch.items()}
        ga_denom = float(micro_batch["loss_mask"].sum().item())

        metrics = alg.train_step(micro_batch, ga_denom=ga_denom, ga_steps=1)
        assert torch.isfinite(torch.tensor(metrics["loss"]))
