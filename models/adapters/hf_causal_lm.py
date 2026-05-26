from __future__ import annotations

from typing import Any, Dict

import torch

from misc.batch_utils import move_to_device
from models.adapters.base import ForwardOutput, ModelAdapter


class HFCausalLMAdapter(ModelAdapter):
    """
    Text-only adapter for HuggingFace causal LMs (AutoModelForCausalLM-style).

    Assumes the batch contains:
      - input_ids:  [B, T]
      - attn_mask:  [B, T]
      - loss_mask:  [B, T-1]
      - position_ids (optional): [B, T]
    """

    def forward(self, model_engine: Any, batch: Dict[str, Any]) -> ForwardOutput:
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        loss_mask = batch["loss_mask"]

        pos_ids = batch.get("position_ids", None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(attn_mask.device)

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=False,
        )

        every_token_logits = outputs.logits  # [B, T, V]
        logits = every_token_logits[:, :-1, :].contiguous()  # [B, T-1, V]
        target_ids = input_ids[:, 1:].contiguous()  # [B, T-1]
        return ForwardOutput(logits=logits, target_ids=target_ids, loss_mask=loss_mask)

    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        return move_to_device(batch, device)

