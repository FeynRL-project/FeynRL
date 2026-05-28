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
                raise ValueError(f"Expected each turn to be a dict, got {type(turn)}: {turn!r}")

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
            audio = mm.get("audio", None)
            if isinstance(audio, dict):
                for k in ("input_features", "feature_attention_mask"):
                    v = audio.get(k, None)
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
        return move_to_device(batch, device)

    def build_multi_modal_inputs(self, processor: Any, mm_items: list[Any]) -> Dict[str, Any]:
        if processor is None:
            raise ValueError("processor is required to build audio inputs for qwen2_audio")
        if not mm_items:
            return {}
        audio_dicts = []
        for m in mm_items:
            if not isinstance(m, dict) or "audio" not in m:
                raise KeyError("Expected each mm_item to be a dict with key 'audio' for qwen2_audio.")
            waveform, sr = m["audio"]
            enc = processor(text=" ", audio=waveform, sampling_rate=int(sr), return_tensors="pt")
            ad: Dict[str, torch.Tensor] = {}
            for k in ("input_features", "feature_attention_mask"):
                if k in enc:
                    ad[k] = enc[k]
            if not ad:
                ad = {k: enc[k] for k in enc.keys() if k not in ("input_ids", "attention_mask")}
            audio_dicts.append(ad)

        keys = set(audio_dicts[0].keys())
        for d in audio_dicts[1:]:
            if set(d.keys()) != keys:
                raise ValueError("Inconsistent audio keys across batch.")
        batched: Dict[str, torch.Tensor] = {}
        for k in keys:
            batched[k] = torch.cat([d[k] for d in audio_dicts], dim=0)
        return {"audio": batched}
