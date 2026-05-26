from __future__ import annotations
from typing import Any, Dict
import torch
from misc.batch_utils import move_to_device
from models.adapters.base import ForwardOutput, ModelAdapter


class Qwen2AudioAdapter(ModelAdapter):
    """
    Adapter for Qwen2-Audio.
    """

    def prepare_messages(self, messages: list) -> list:
        out = []
        injected = False
        for turn in messages:
            if not isinstance(turn, dict):
                continue

            role = turn.get("role", None)
            content = turn.get("content", "")
            new_turn = dict(turn)

            if (not injected) and role == "user" and isinstance(content, str):
                new_turn["content"] = [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": content},
                ]
                injected = True
            elif isinstance(content, str):
                new_turn["content"] = [{"type": "text", "text": content}]

            out.append(new_turn)
        return out

    def get_mm_kwargs(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        mm = batch.get("multi_modal_inputs", None)
        if not isinstance(mm, dict):
            return {}
        audio = mm.get("audio", None)
        if not isinstance(audio, dict):
            return {}
        mm_kwargs: Dict[str, Any] = {}
        for k in ("input_features", "feature_attention_mask"):
            v = audio.get(k, None)
            if v is not None and torch.is_tensor(v):
                mm_kwargs[k] = v
        return mm_kwargs

    def forward(self, model_engine: Any, batch: Dict[str, Any]) -> ForwardOutput:
        input_ids = batch["input_ids"]
        attn_mask = batch["attn_mask"]
        loss_mask = batch["loss_mask"]

        pos_ids = batch.get("position_ids", None)
        if pos_ids is not None:
            pos_ids = pos_ids.to(attn_mask.device)

        mm_kwargs = self.get_mm_kwargs(batch)

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
        return move_to_device(batch, device)
