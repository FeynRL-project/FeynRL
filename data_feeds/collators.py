from __future__ import annotations

from typing import Any, Dict, List

import torch


def build_preference_multimodal_collate_fn(enable_vision: bool = True) -> Any:
    """
    Collate function for DPO-style preference batches.

    Per-sample expected keys:
      - input_ids:  [2, T]
      - attn_mask:  [2, T]
      - loss_mask:  [2, T-1]
      - multi_modal_inputs (optional): {"vision": {...}} where tensors are prompt-level

    Batch output:
      - input_ids: [B, 2, T]
      - attn_mask: [B, 2, T]
      - loss_mask: [B, 2, T-1]
      - multi_modal_inputs["vision"]: [2B, ...] (duplicated to match DPO flattening)
    """

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {}

        out: Dict[str, Any] = {
            "input_ids": torch.stack([s["input_ids"] for s in batch], dim=0),
            "attn_mask": torch.stack([s["attn_mask"] for s in batch], dim=0),
            "loss_mask": torch.stack([s["loss_mask"] for s in batch], dim=0),
        }

        if enable_vision:
            vision_items = []
            for s in batch:
                mm = s.get("multi_modal_inputs") or {}
                vision = mm.get("vision")
                if vision is None:
                    raise KeyError("Missing multi_modal_inputs['vision'] for a multimodal preference batch")
                vision_items.append(vision)

            keys = set(vision_items[0].keys())
            for v in vision_items[1:]:
                if set(v.keys()) != keys:
                    raise ValueError("Inconsistent vision tensor keys across batch")

            vision_batched: Dict[str, torch.Tensor] = {}
            for k in keys:
                vision_batched[k] = torch.cat([v[k] for v in vision_items], dim=0)

            # Duplicate prompt-level vision tensors so they align with DPO's
            # [B,2,T] -> [2B,T] flattening.
            for k, t in vision_batched.items():
                vision_batched[k] = torch.cat([t, t], dim=0)

            out["multi_modal_inputs"] = {"vision": vision_batched}

        return out

    return collate

