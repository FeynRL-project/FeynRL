from __future__ import annotations
import inspect
import io
import os
from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch
from datasets import load_dataset


def _get_audio_kwarg(processor: Any) -> str:
    """Return the kwarg name this processor uses for the audio waveform."""
    try:
        params = inspect.signature(processor.__call__).parameters
    except (ValueError, TypeError):
        return "audios"
    for name in ("audios", "audio", "raw_speech"):
        if name in params:
            return name
    return "audios"


def _load_audio_bytes(payload: Any) -> Tuple[np.ndarray, Optional[int]]:
    """Decode audio bytes to (waveform_float32, sampling_rate)."""
    if isinstance(payload, np.ndarray):
        return payload.astype(np.float32), None
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32), None
    if isinstance(payload, (bytes, bytearray)):
        try:
            import soundfile as sf  # type: ignore
        except Exception as e:
            raise ImportError(
                "soundfile is required to decode audio_bytes. Install it via `pip install soundfile`."
            ) from e
        waveform, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        return np.asarray(waveform, dtype=np.float32), int(sr)
    raise TypeError(f"Unsupported audio payload type: {type(payload)}")


class AudioPairedFeed:
    """
    SFT dataset for audio+text tasks (single prompt → single answer).

    Expected parquet columns:
      - prompt:       list[{role, content}]
      - answer:       str
      - audio_bytes:  bytes (encoded audio) or float array

    Returns per item:
      - input_ids:          [T]
      - attn_mask:          [T]
      - loss_mask:          [T-1]
      - multi_modal_inputs: {"audio": {input_features, feature_attention_mask}}
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
        audio_key: str = "audio_bytes",
        sampling_rate_key: str = "sampling_rate",
        default_sampling_rate: int = 16000,
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
        self.audio_key = audio_key
        self.sampling_rate_key = sampling_rate_key
        self.default_sampling_rate = int(default_sampling_rate)
        self._audio_kwarg = _get_audio_kwarg(processor)
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
        self, messages: list, answer: str, waveform, sr: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        template_fn = getattr(self.processor, "apply_chat_template", None) or self.tokenizer.apply_chat_template
        prompt_text = template_fn(messages, add_generation_prompt=True, tokenize=False)
        eos = getattr(self.tokenizer, "eos_token", None) or ""
        full_text = prompt_text + str(answer) + eos

        audio_kw = {self._audio_kwarg: waveform, "sampling_rate": sr}

        # True prompt length including audio feature tokens injected by the processor.
        prompt_len = self.processor(text=prompt_text, return_tensors="pt", **audio_kw)["input_ids"].shape[1]

        enc = self.processor(
            text=full_text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            **audio_kw,
        )
        input_ids = enc["input_ids"][0]
        attn_mask = enc["attention_mask"][0]
        loss_mask = attn_mask[1:].clone()
        if prompt_len > 1:
            loss_mask[:prompt_len - 1] = 0

        audio_dict: Dict[str, torch.Tensor] = {}
        for k in ("input_features", "feature_attention_mask"):
            if k in enc:
                audio_dict[k] = enc[k]

        return input_ids, attn_mask, loss_mask, audio_dict

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        idx = int(idx)
        sample = self.data[idx]
        messages = sample[self.prompt_key]
        answer = sample[self.answer_key]

        audio_payload = sample.get(self.audio_key)
        if audio_payload is None:
            raise KeyError(f"Missing '{self.audio_key}' in sample keys={list(sample.keys())}")

        sr_from_sample = sample.get(self.sampling_rate_key, None)
        default_sr = int(sr_from_sample or self.default_sampling_rate)
        waveform, file_sr = _load_audio_bytes(audio_payload)
        sr = file_sr or default_sr

        if self.adapter is not None and hasattr(self.adapter, "prepare_messages"):
            messages = self.adapter.prepare_messages(messages)

        input_ids, attn_mask, loss_mask, audio_dict = self._encode(messages, answer, waveform, sr)

        if loss_mask.sum().item() == 0:
            raise ValueError(f"Sample {idx}: no training tokens after masking")

        return {
            "input_ids": input_ids,
            "attn_mask": attn_mask,
            "loss_mask": loss_mask,
            "multi_modal_inputs": {"audio": audio_dict},
        }
