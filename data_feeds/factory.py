from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch

from models.adapters import get_adapter


# ---------------------------------------------------------------------------
# Preference / DPO collate (previously data_feeds/collators.py)
# ---------------------------------------------------------------------------

def _preference_vision_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate for DPO image-preference batches.

    Per-sample keys: input_ids [2,T], attn_mask [2,T], loss_mask [2,T-1],
                     multi_modal_inputs["vision"] (prompt-level tensors).
    Batch output stacks those and duplicates vision to match DPO's [2B] flattening.
    """
    if not batch:
        return {}

    out: Dict[str, Any] = {
        "input_ids": torch.stack([s["input_ids"] for s in batch], dim=0),
        "attn_mask": torch.stack([s["attn_mask"] for s in batch], dim=0),
        "loss_mask": torch.stack([s["loss_mask"] for s in batch], dim=0),
    }

    vision_items = []
    for s in batch:
        mm = s.get("multi_modal_inputs") or {}
        vision = mm.get("vision")
        if vision is None:
            raise KeyError("Missing multi_modal_inputs['vision'] in multimodal preference batch")
        vision_items.append(vision)

    keys = set(vision_items[0].keys())
    for v in vision_items[1:]:
        if set(v.keys()) != keys:
            raise ValueError("Inconsistent vision tensor keys across batch")

    vision_batched: Dict[str, torch.Tensor] = {}
    for k in keys:
        vision_batched[k] = torch.cat([v[k] for v in vision_items], dim=0)

    # Duplicate so vision aligns with DPO's [B,2,T] -> [2B,T] flattening.
    for k, t in vision_batched.items():
        vision_batched[k] = torch.cat([t, t], dim=0)

    out["multi_modal_inputs"] = {"vision": vision_batched}
    return out


# ---------------------------------------------------------------------------
# Feed factories
# ---------------------------------------------------------------------------

def make_sft_feed(
    model_class: str | None,
    params: Any,
    processor: Any = None,
) -> Tuple[Type, Dict[str, Any]]:
    """Return (dataset_cls, dataset_kwargs) for SFT (paired) data loaders."""
    from data_feeds.paired import PairedFeed
    mc = model_class or ""
    if mc == "qwen2_5_vl":
        from data_feeds.image_paired import ImagePairedFeed
        return ImagePairedFeed, {
            "processor": processor,
            "adapter": get_adapter(mc),
            "image_bytes_key": getattr(params.data, "image_bytes_key", None) or "image_bytes",
            "image_placeholder_token": getattr(params.data, "image_placeholder_token", None) or "<image>",
            "insert_image_token_if_missing": bool(getattr(params.data, "insert_image_token_if_missing", False)),
            "max_image_pixels": getattr(params.data, "max_image_pixels", None),
        }
    if mc == "qwen2_audio":
        from data_feeds.audio_paired import AudioPairedFeed
        return AudioPairedFeed, {
            "processor": processor,
            "adapter": get_adapter(mc),
            "audio_key": getattr(params.data, "audio_key", None) or "audio_bytes",
            "sampling_rate_key": getattr(params.data, "sampling_rate_key", None) or "sampling_rate",
            "default_sampling_rate": getattr(params.data, "default_sampling_rate", None) or 16000,
        }
    return PairedFeed, {}


def make_preference_feed(
    model_class: str | None,
    params: Any,
    processor: Any = None,
) -> Tuple[Type, Dict[str, Any], Optional[Callable]]:
    """Return (dataset_cls, dataset_kwargs, collate_fn) for preference (DPO/CL) data loaders."""
    from data_feeds.preference import PreferenceFeed
    mc = model_class or ""
    if mc == "qwen2_5_vl":
        from data_feeds.image_preference import ImagePreferenceFeed
        return ImagePreferenceFeed, {
            "processor": processor,
            "adapter": get_adapter(mc),
            "image_bytes_key": getattr(params.data, "image_bytes_key", None) or "image_bytes",
            "image_placeholder_token": getattr(params.data, "image_placeholder_token", None) or "<image>",
            "insert_image_token_if_missing": bool(getattr(params.data, "insert_image_token_if_missing", False)),
        }, _preference_vision_collate
    return PreferenceFeed, {}, None


def make_rollout_feed(
    model_class: str | None,
    params: Any,
    processor: Any = None,
) -> Tuple[Type, Dict[str, Any]]:
    """Return (dataset_cls, dataset_kwargs) for RL rollout (prompt) data loaders."""
    from data_feeds.prompts import PromptsFeed
    mc = model_class or ""
    if mc == "qwen2_5_vl":
        from data_feeds.image_prompts import ImagePromptsFeed
        return ImagePromptsFeed, {
            "adapter": get_adapter(mc),
            "image_key": getattr(params.data, "image_bytes_key", None) or "image_bytes",
            "max_image_pixels": getattr(params.data, "max_image_pixels", None),
        }
    if mc == "qwen2_audio":
        from data_feeds.audio_prompts import AudioPromptsFeed
        if processor is None:
            from transformers import AutoProcessor
            proc_name = getattr(params.model, "processor_name_or_path", None) or params.model.name
            processor = AutoProcessor.from_pretrained(
                proc_name, trust_remote_code=params.model.trust_remote_code
            )
        return AudioPromptsFeed, {
            "adapter": get_adapter(mc),
            "audio_key": getattr(params.data, "audio_key", None) or "audio_bytes",
            "sampling_rate_key": getattr(params.data, "sampling_rate_key", None) or "sampling_rate",
            "default_sampling_rate": getattr(params.data, "default_sampling_rate", None) or 16000,
            "processor": processor,
        }
    return PromptsFeed, {}
