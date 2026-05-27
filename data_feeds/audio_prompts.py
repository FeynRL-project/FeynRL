from __future__ import annotations
import io
from typing import Any, Dict, Optional, Tuple
import numpy as np
from data_feeds.prompts import PromptsFeed


def _load_audio_bytes(payload: Any) -> Tuple[np.ndarray, Optional[int]]:
    """
    Decode audio bytes to (waveform_float32, sampling_rate).
    Supports encoded audio files (WAV, FLAC, etc.) via soundfile.
    """
    if isinstance(payload, np.ndarray):
        return payload.astype(np.float32), None
    if isinstance(payload, list):
        return np.asarray(payload, dtype=np.float32), None

    if isinstance(payload, (bytes, bytearray)):
        try:
            import soundfile as sf  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "soundfile is required to decode audio_bytes. Install it via `pip install soundfile`."
            ) from e

        waveform, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        return np.asarray(waveform, dtype=np.float32), int(sr)

    raise TypeError(f"Unsupported audio payload type: {type(payload)}")


class AudioPromptsFeed(PromptsFeed):
    """
    RL rollout data feed for audio+text tasks.

    Returns a vLLM-ready multimodal dict so rollout engines remain modality-agnostic.
    Expected parquet columns:
      - prompt:       list[{role, content}]
      - solution:     str | list[str] (optional)
      - audio_bytes:  bytes (encoded audio file) OR float array/list
      - sampling_rate: int (optional; otherwise derived from file)
    """

    def __init__(
        self,
        prompt_key: str,
        tokenizer: Any,
        max_seq_len: int,
        data_path: str,
        solution_key: str | None = None,
        audio_key: str = "audio_bytes",
        sampling_rate_key: str = "sampling_rate",
        default_sampling_rate: int = 16000,
        adapter: Any | None = None,
        processor: Any | None = None,
    ):
        super().__init__(
            prompt_key=prompt_key,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            data_path=data_path,
            solution_key=solution_key,
        )
        self.audio_key = audio_key
        self.sampling_rate_key = sampling_rate_key
        self.default_sampling_rate = int(default_sampling_rate)
        self.adapter = adapter
        self.processor = processor

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data[int(idx)]
        if self.prompt_key not in sample:
            raise KeyError(f"Missing key '{self.prompt_key}' in sample: keys={list(sample.keys())}")

        messages = sample[self.prompt_key]
        if not messages or (isinstance(messages, list) and len(messages) == 0):
            raise ValueError(f"Sample {idx}: Prompt cannot be empty")

        if self.adapter is not None:
            messages = self.adapter.prepare_messages(messages)

        template_fn = getattr(self.processor, "apply_chat_template", None) or self.tokenizer.apply_chat_template
        prompt_text = template_fn(messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer.encode(prompt_text)

        if not isinstance(prompt_ids, list) or len(prompt_ids) == 0:
            raise ValueError(f"Sample {idx}: tokenization produced empty prompt_ids")
        if len(prompt_ids) >= self.max_seq_len:
            raise ValueError(f"Prompt in sample {idx} too long: {len(prompt_ids)} tokens")

        audio_payload = sample.get(self.audio_key, None)
        if audio_payload is None:
            raise KeyError(f"Missing audio payload key '{self.audio_key}' in sample {idx}: keys={list(sample.keys())}")

        waveform, file_sr = _load_audio_bytes(audio_payload)
        sr = file_sr or int(sample.get(self.sampling_rate_key, self.default_sampling_rate) or self.default_sampling_rate)

        out: Dict[str, Any] = {"prompt": prompt_text, "multi_modal_data": {"audio": (waveform, sr)}}
        if self.solution_key:
            out["solution"] = sample[self.solution_key]
        return out
