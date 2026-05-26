from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import os

import torch
from datasets import load_dataset
from PIL import Image


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


class ImagePreferenceFeed:
    """
    Preference dataset for image-conditioned DPO.

    Output per item:
      - input_ids: [2, T]  (row 0 chosen, row 1 rejected)
      - attn_mask: [2, T]
      - loss_mask: [2, T-1]
      - multi_modal_inputs: {"vision": { ... }}  (batched later by collator)
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
    ):
        assert prompt_key
        assert answer_key
        assert max_seq_len > 0
        assert tokenizer is not None
        assert processor is not None
        assert os.path.exists(os.path.expanduser(data_path)), f"{data_path} does not exist"

        self.prompt_key = prompt_key
        self.chosen_key = answer_key
        self.rejected_key = "rejected_" + answer_key
        self.max_seq_len = int(max_seq_len)
        self.tokenizer = tokenizer
        self.processor = processor
        self.adapter = adapter
        self.data_path = data_path
        self.image_bytes_key = image_bytes_key
        self.image_placeholder_token = image_placeholder_token
        self.insert_image_token_if_missing = bool(insert_image_token_if_missing)
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

    def _encode(self, messages: list, answer: str, pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        # Render prompt with chat template and append answer + eos.
        try:
            prompt_text = self.tokenizer.apply_chat_template(
                conversation=messages,
                add_generation_prompt=True,
                tokenize=False,
                skip_special_tokens=False,
            )
        except TypeError:
            # Some processors/tokenizers accept positional `messages` signature.
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )
        eos = getattr(self.tokenizer, "eos_token", None) or ""
        full_text = prompt_text + str(answer) + eos

        enc = self.processor(
            text=full_text,
            images=pil,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
        )
        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]
        # loss_mask: compute loss on all non-pad tokens except the first token (shifted labels)
        loss_mask = attn_mask[1:].clone()

        mm_dict: Dict[str, torch.Tensor] = {}
        for k in ("pixel_values", "image_grid_thw"):
            if k in enc:
                mm_dict[k] = enc[k]
        if not mm_dict:
            # Fallback: include any non-text keys as vision tensors.
            for k, v in enc.items():
                if k not in ("input_ids", "attention_mask"):
                    mm_dict[k] = v

        return input_ids, attn_mask, mm_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = int(idx)
        sample = self.data[idx]
        messages = sample[self.prompt_key]
        chosen = sample[self.chosen_key]
        rejected = sample[self.rejected_key]

        image_payload = sample.get(self.image_bytes_key)
        if image_payload is None:
            raise KeyError(f"Missing '{self.image_bytes_key}' in sample keys={list(sample.keys())}")
        pil = _load_pil_image(image_payload)

        if self.adapter is not None and hasattr(self.adapter, "prepare_messages"):
            messages = self.adapter.prepare_messages(messages)
        else:
            messages = _ensure_image_token(messages, self.image_placeholder_token, self.insert_image_token_if_missing)

        chosen_ids, chosen_mask, vision = self._encode(messages, chosen, pil)
        rejected_ids, rejected_mask, _vision2 = self._encode(messages, rejected, pil)

        # Stack into [2, T]
        input_ids = torch.stack([chosen_ids, rejected_ids], dim=0)
        attn_mask = torch.stack([chosen_mask, rejected_mask], dim=0)
        loss_mask = torch.stack([chosen_mask[1:].clone(), rejected_mask[1:].clone()], dim=0)

        return {
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "loss_mask": loss_mask,
            "multi_modal_inputs": {"vision": vision},
        }
