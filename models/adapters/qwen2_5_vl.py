from __future__ import annotations
from typing import Any, Dict
import torch
from misc.batch_utils import move_to_device
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
                pv = vision.get("pixel_values", None)
                if pv is not None and torch.is_tensor(pv):
                    # DataLoader stacks [N_patches, D] per sample into [B, N_patches, D].
                    # The visual encoder expects [B*N_patches, D].
                    if pv.ndim == 3:
                        pv = pv.flatten(0, 1)
                    mm_kwargs["pixel_values"] = pv

                thw = vision.get("image_grid_thw", None)
                if thw is not None and torch.is_tensor(thw):
                    # DataLoader stacks [N_imgs, 3] per sample into [B, N_imgs, 3].
                    # The visual encoder expects [B*N_imgs, 3].
                    if thw.ndim == 3:
                        thw = thw.flatten(0, 1)
                    mm_kwargs["image_grid_thw"] = thw

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
            raise ValueError("processor is required to build vision inputs for qwen2_5_vl")
        if not mm_items:
            return {}
        vision_dicts = []
        for m in mm_items:
            if not isinstance(m, dict) or "image" not in m:
                raise KeyError("Expected each mm_item to be a dict with key 'image' for qwen2_5_vl.")
            enc = processor(text=" ", images=m["image"], return_tensors="pt")
            if "pixel_values" in enc or "image_grid_thw" in enc:
                vd = {k: enc[k] for k in ("pixel_values", "image_grid_thw") if k in enc}
            else:
                vd = {k: enc[k] for k in enc.keys() if k not in ("input_ids", "attention_mask")}
            vision_dicts.append(vd)

        keys = set(vision_dicts[0].keys())
        for d in vision_dicts[1:]:
            if set(d.keys()) != keys:
                raise ValueError("Inconsistent vision keys across batch.")
        batched: Dict[str, torch.Tensor] = {}
        for k in keys:
            batched[k] = torch.cat([d[k] for d in vision_dicts], dim=0)
        return {"vision": batched}
