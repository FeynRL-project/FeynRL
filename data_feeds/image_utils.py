from __future__ import annotations

import os
from io import BytesIO
from typing import Any, Optional

import PIL.PngImagePlugin
from PIL import Image

# Some PNGs embed large ICC profiles that exceed PIL's default 1 MB cap.
PIL.PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024


def _load_pil_image(payload: Any) -> Image.Image:
    if isinstance(payload, Image.Image):
        return payload.convert("RGB")
    if isinstance(payload, (bytes, bytearray)):
        return Image.open(BytesIO(payload)).convert("RGB")
    if isinstance(payload, str):
        path = os.path.expanduser(payload)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image path does not exist: {payload}")
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image payload type: {type(payload)}")


def _ensure_image_token(messages: list, placeholder: str, insert_if_missing: bool) -> list:
    if not insert_if_missing:
        return messages
    out = []
    injected = False
    for turn in messages:
        if (
            not injected
            and isinstance(turn, dict)
            and turn.get("role") == "user"
            and isinstance(turn.get("content"), str)
            and placeholder not in turn.get("content", "")
        ):
            new_turn = dict(turn)
            new_turn["content"] = placeholder + new_turn["content"]
            out.append(new_turn)
            injected = True
        else:
            out.append(turn)
    return out


def _load_and_resize(image_bytes: Any, max_image_pixels: Optional[int] = None) -> Image.Image:
    pil = _load_pil_image(image_bytes)
    if max_image_pixels is None:
        return pil
    w, h = pil.size
    if w * h <= int(max_image_pixels):
        return pil
    scale = (float(max_image_pixels) / float(w * h)) ** 0.5
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return pil.resize((new_w, new_h), Image.LANCZOS)
