import torch
from unittest.mock import MagicMock
from algs.DPO.dpo import DPO
from data_feeds.factory import _preference_vision_collate
from models.adapters.base import ForwardOutput


def test_preference_multimodal_collate_duplicates_vision():
    T = 6
    samples = []
    for _ in range(2):
        samples.append(
            {
                "input_ids": torch.zeros(2, T, dtype=torch.long),
                "attn_mask": torch.ones(2, T, dtype=torch.long),
                "loss_mask": torch.ones(2, T - 1, dtype=torch.long),
                "multi_modal_inputs": {
                    "vision": {
                        "pixel_values": torch.randn(1, 3, 2, 2),
                        "image_grid_thw": torch.ones(1, 3, dtype=torch.long),
                    }
                },
            }
        )

    batch = _preference_vision_collate(samples)

    assert batch["input_ids"].shape == (2, 2, T)
    assert batch["attn_mask"].shape == (2, 2, T)
    assert batch["loss_mask"].shape == (2, 2, T - 1)

    vision = batch["multi_modal_inputs"]["vision"]
    assert vision["pixel_values"].shape[0] == 4  # 2B after duplication
    assert vision["image_grid_thw"].shape[0] == 4


def test_dpo_forward_with_adapter_uses_multimodal_batch():
    class DummyAdapter:
        def __init__(self):
            self.calls = []

        def forward(self, model_engine, batch):
            self.calls.append(model_engine)
            input_ids = batch["input_ids"]
            loss_mask = batch["loss_mask"]
            B, T = input_ids.shape
            V = 7
            logits = torch.zeros(B, T - 1, V)
            target_ids = input_ids[:, 1:].contiguous()
            return ForwardOutput(logits=logits, target_ids=target_ids, loss_mask=loss_mask)

    model_engine = MagicMock()
    ref_model_engine = MagicMock()
    optimizer = MagicMock()

    adapter = DummyAdapter()
    dpo = DPO(
        model_engine=model_engine,
        ref_model_engine=ref_model_engine,
        optimizer=optimizer,
        beta=0.1,
        model_adapter=adapter,
    )

    T = 5
    batch = {
        "input_ids": torch.zeros(2, 2, T, dtype=torch.long),
        "attn_mask": torch.ones(2, 2, T, dtype=torch.long),
        "loss_mask": torch.ones(2, 2, T - 1, dtype=torch.long),
        "multi_modal_inputs": {"vision": {"pixel_values": torch.randn(4, 3, 2, 2)}},
    }

    logprobs, ref_logprobs, loss_mask = dpo.forward(batch)
    assert logprobs.shape == (4, T - 1)
    assert ref_logprobs.shape == (4, T - 1)
    assert loss_mask.shape == (4, T - 1)
    assert adapter.calls == [ref_model_engine, model_engine]
