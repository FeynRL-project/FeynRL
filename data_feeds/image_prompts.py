from __future__ import annotations

from typing import Any, Dict, Optional

from PIL import Image

from data_feeds.prompts import PromptsFeed
from data_feeds.image_utils import _load_and_resize


class ImagePromptsFeed(PromptsFeed):
    """
    RL rollout data feed for image+text tasks.

    Returns a vLLM-ready multimodal dict so rollout engines remain modality-agnostic.
    Expected parquet columns:
      - prompt:       list[{role, content}]
      - solution:     str (optional)
      - image_bytes:  bytes (PNG/JPEG-encoded image)
    """

    def __init__(
        self,
        prompt_key: str,
        tokenizer: Any,
        max_seq_len: int,
        data_path: str,
        solution_key: str | None = None,
        image_key: str = "image_bytes",
        adapter: Any | None = None,
        max_image_pixels: int | None = None,
    ):
        super().__init__(
            prompt_key=prompt_key,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            data_path=data_path,
            solution_key=solution_key,
        )
        self.image_key = image_key
        self.adapter = adapter
        self.max_image_pixels = max_image_pixels

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[int(idx)]
        if self.prompt_key not in sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample: keys={list(sample.keys())}")

        messages = sample[self.prompt_key]
        if not messages or (isinstance(messages, list) and len(messages) == 0):
            raise ValueError(f"Sample {idx}: Prompt cannot be empty")

        if self.adapter is not None:
            messages = self.adapter.prepare_messages(messages)

        prompt_text = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False,
            skip_special_tokens=False,
        )

        prompt_ids = self.tokenizer.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
            raise ValueError(f"Sample {idx}: tokenization produced empty prompt_ids")
        if len(prompt_ids) >= self.max_seq_len:
            raise ValueError(f"Prompt in sample {idx} too long: {len(prompt_ids)} tokens")

        image_bytes = sample.get(self.image_key, None)
        if image_bytes is None:
            raise KeyError(f"Missing image payload key '{self.image_key}' in sample {idx}: keys={list(sample.keys())}")

        pil = _load_and_resize(image_bytes, self.max_image_pixels)
        out: Dict[str, Any] = {"prompt": prompt_text, "multi_modal_data": {"image": pil}}
        if self.solution_key:
            out["solution"] = sample[self.solution_key]
        return out

