from __future__ import annotations

from typing import Any, Dict

import torch

from models.adapters.base import ForwardOutput, ModelAdapter


class Qwen2_5VLAdapter(ModelAdapter):
    """
    Adapter for Qwen2.5-VL.
    """

    def prepare_messages(self, messages: list) -> list:
        image_token = "<|vision_start|><|image_pad|><|vision_end|>"
        out = []
        injected = False
        for turn in messages:
            if (
                not injected
                and isinstance(turn, dict)
                and turn.get("role") == "user"
                and isinstance(turn.get("content", None), str)
                and image_token not in str(turn.get("content", ""))
            ):
                new_turn = dict(turn)
                new_turn["content"] = image_token + new_turn["content"]
                turn = new_turn
                injected = True
            out.append(turn)
        return out

    def forward(self, model_engine: Any, batch: Dict[str, Any]) -> ForwardOutput:
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        loss_mask = batch["loss_mask"]

        pos_ids = batch.get("position_ids", None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(attn_mask.device)

        mm = batch.get("multi_modal_inputs", None)
        mm_kwargs: Dict[str, Any] = {}
        if isinstance(mm, dict):
            vision = mm.get("vision", None)
            if isinstance(vision, dict):
                for k in ("pixel_values", "image_grid_thw"):
                    v = vision.get(k, None)
                    if v is not None and torch.is_tensor(v):
                        mm_kwargs[k] = v

        outputs = model_engine(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=pos_ids,
            use_cache=False,
            **mm_kwargs,
        )

        logits = outputs.logits[:, :-1, :].contiguous()
        target_ids = input_ids[:, 1:].contiguous()
        return ForwardOutput(logits=logits, target_ids=target_ids, loss_mask=loss_mask)

    def to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in batch.items():
            out[k] = v.to(device) if torch.is_tensor(v) else v
        return out

