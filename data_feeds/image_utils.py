import io
import os
from PIL import Image

def load_image(obj):
    '''
        Decode a single image from a variety of storage formats such as 
        raw bytes (i.e. PIL.Image.Image), a HF Image dict ({'bytes':..., 'path':...}), 
        a file path (i.e., file path), or an already-decoded PIL image, 
        and return a normalized RGB PIL image
    '''
    if isinstance(obj, Image.Image):
        img = obj

    elif isinstance(obj, dict):
        if obj.get("bytes") is not None:
            img = Image.open(io.BytesIO(obj["bytes"]))
        elif obj.get("path"):
            img = Image.open(os.path.expanduser(obj["path"]))
        else:
            raise ValueError(f"Image dict has neither 'bytes' nor 'path': keys={list(obj.keys())}")

    elif isinstance(obj, (bytes, bytearray)):
        img = Image.open(io.BytesIO(obj))

    elif isinstance(obj, str):
        img = Image.open(os.path.expanduser(obj))

    else:
        raise TypeError(f"Unsupported image type {type(obj)}; expected PIL.Image, dict, bytes, or path str")

    # convert handles palette/alpha/ICC-profile inputs into a plain 3-channel RGB image.
    return img.convert("RGB")

def resize(img, max_pixels):
    '''
        Downscale img (preserving aspect ratio) so width*height <= max_pixels.
        This bounds how many image tokens the processor produces, which keeps prompts
        within max_seq_len and avoids truncation through an image span. No-op when
        max_pixels is None or the image is already small enough.
    '''
    if max_pixels is None:
        return img

    w, h = img.size
    if w * h <= max_pixels:
        return img

    scale = (max_pixels / float(w * h)) ** 0.5
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return img.resize(new_size, Image.BICUBIC)

def load_images(raw, max_pixels=None):
    '''
        Normalize a parquet image cell into a list of RGB PIL images (capped to
        max_pixels). raw may be a single image (any form load_image accepts) or a
        list of them. Returns [] when raw is None/empty.
    '''
    if raw is None:
        return []
    if not isinstance(raw, list):
        raw = [raw]
    return [resize(load_image(r), max_pixels) for r in raw]