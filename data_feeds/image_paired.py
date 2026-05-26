from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import torch
from datasets import load_dataset
from PIL import Image

from data_feeds.image_preference import _ensure_image_token, _load_pil_image


class ImagePairedFeed:
    """
    SFT dataset for image+text tasks (single prompt → single answer).

    Expected parquet columns:
      - prompt:       list[{role, content}]
      - answer:       str
      - image_bytes:  bytes (PNG/JPEG-encoded image)

    Returns per item:
      - input_ids:          [T]
      - attn_mask:          [T]
      - loss_mask:          [T-1]  (all non-padding tokens; answer + prompt)
      - multi_modal_inputs: {"vision": {pixel_values, image_grid_thw, ...}}
    """

    def __init__(
        self,
        prompt_key: str,
        answer_key: str,
        max_seq_len: int,
        tokenizer: Any,
        data_path: str,
        processor: Any,
        adapter: Any = None,
        image_bytes_key: str = "image_bytes",
        image_placeholder_token: str = "<image>",
        insert_image_token_if_missing: bool = False,
        max_image_pixels: int | None = None,
    ):
        assert prompt_key
        assert answer_key
        assert max_seq_len > 0
        assert tokenizer is not None
        assert processor is not None
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.max_seq_len = int(max_seq_len)
        self.tokenizer = tokenizer
        self.processor = processor
        self.adapter = adapter
        self.data_path = data_path
        self.image_bytes_key = image_bytes_key
        self.image_placeholder_token = image_placeholder_token
        self.insert_image_token_if_missing = bool(insert_image_token_if_missing)

        # Each Qwen2.5-VL visual token covers a 28×28-pixel block (14-px patch × merge_size 2).
        # Cap the processor's image resolution so the visual tokens leave room for text.
        effective = max_image_pixels if max_image_pixels is not None else (self.max_seq_len - 256) * 28 * 28
        img_proc = getattr(processor, "image_processor", None)
        if img_proc is not None and hasattr(img_proc, "max_pixels"):
            img_proc.max_pixels = effective

        self._load_data()

    def _load_data(self) -> None:
        try:
            self.data = load_dataset("parquet", data_files=self.data_path, split="train")
        except PermissionError:
            cache_dir = os.environ.get("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
            os.makedirs(cache_dir, exist_ok=True)
            self.data = load_dataset("parquet", data_files=self.data_path, split="train", cache_dir=cache_dir)
        self.len_data = len(self.data)

    def __len__(self) -> int:
        return self.len_data

    def _encode(
        self, messages: list, answer: str, pil: Image.Image
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        template_fn = getattr(self.processor, "apply_chat_template", None) or self.tokenizer.apply_chat_template
        prompt_text = template_fn(
            messages, add_generation_prompt=True, tokenize=False
        )

        eos = getattr(self.tokenizer, "eos_token", None) or ""
        full_text = prompt_text + str(answer) + eos

        enc = self.processor(
            text=full_text,
            images=pil,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]

        T = input_ids.shape[0]
        if T > self.max_seq_len:
            raise ValueError(f"Sample too long after image resizing: {T} tokens > max_seq_len {self.max_seq_len}")

        pad_len = self.max_seq_len - T
        if pad_len > 0:
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.cat([input_ids, input_ids.new_full((pad_len,), pad_id)])
            attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(pad_len)])

        loss_mask = attn_mask[1:].clone()

        mm_dict: Dict[str, torch.Tensor] = {}
        for k in ("pixel_values", "image_grid_thw"):
            if k in enc:
                mm_dict[k] = enc[k]
        if not mm_dict:
            for k, v in enc.items():
                if k not in ("input_ids", "attention_mask"):
                    mm_dict[k] = v

        return input_ids, attn_mask, loss_mask, mm_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = int(idx)
        sample = self.data[idx]
        messages = sample[self.prompt_key]
        answer = sample[self.answer_key]

        image_payload = sample.get(self.image_bytes_key)
        if image_payload is None:
            raise KeyError(f"Missing '{self.image_bytes_key}' in sample keys={list(sample.keys())}")
        pil = _load_pil_image(image_payload)

        if self.adapter is not None and hasattr(self.adapter, "prepare_messages"):
            messages = self.adapter.prepare_messages(messages)
        else:
            messages = _ensure_image_token(messages, self.image_placeholder_token, self.insert_image_token_if_missing)

        input_ids, attn_mask, loss_mask, vision = self._encode(messages, answer, pil)

        if loss_mask.sum().item() == 0:
            raise ValueError(f"Sample {idx}: no training tokens after masking")

        return {
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "loss_mask": loss_mask,
            "multi_modal_inputs": {"vision": vision},
        }
