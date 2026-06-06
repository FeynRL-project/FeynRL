from __future__ import annotations
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import torch
from models.adapters import get_adapter
import models

# ---------------------------------------------------------------------------
# Vision collate helpers
#
# PyTorch's default collate uses torch.stack, which requires identical shapes.
# pixel_values is [N_patches, D] where N_patches varies by image aspect ratio,
# so we must torch.cat along dim=0 instead.  Both SFT and DPO VLM feeds need
# this; the DPO collate additionally interleaves vision to match its [2B] layout.
# ---------------------------------------------------------------------------

def _vision_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate for SFT image batches (ImagePairedFeed)."""
    out: Dict[str, Any] = {
        "input_ids": torch.stack([s["input_ids"] for s in batch]),
        "attn_mask": torch.stack([s["attn_mask"] for s in batch]),
        "loss_mask": torch.stack([s["loss_mask"] for s in batch]),
    }
    visions = [((s.get("multi_modal_inputs") or {}).get("vision") or {}) for s in batch]
    if any(visions):
        first = next(v for v in visions if v)
        keys = set(first.keys())
        for v in visions:
            if set(v.keys()) != keys:
                raise ValueError("Inconsistent vision tensor keys across batch")
        out["multi_modal_inputs"] = {"vision": {k: torch.cat([v[k] for v in visions], dim=0) for k in keys}}

    return out

# ---------------------------------------------------------------------------
# Feed factories
# ---------------------------------------------------------------------------

def make_sft_feed(
    model_class: str | None,
    params: Any,
    processor: Any = None,
) -> Tuple[Type, Dict[str, Any], Optional[Callable]]:
    """Return (dataset_cls, dataset_kwargs, collate_fn) for SFT (paired) data loaders."""
    from data_feeds.paired import PairedFeed
    mc = model_class or ""
    if mc == "qwen2_5_vl":
        from data_feeds.image_paired import ImagePairedFeed
        return ImagePairedFeed, {
            "processor": processor,
            "adapter": get_adapter(mc),
            "max_image_pixels": getattr(params.data, "max_image_pixels", None),
        }, _vision_collate
    return PairedFeed, {}, None
