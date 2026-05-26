"""ASR reward for LibriSpeech-style transcripts.

Computes a scalar reward from word error rate (WER) between the model's
transcription and the reference transcript(s).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

import torch


_PUNCT_RE = re.compile(r"[^\w\s]")
_SPACE_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACE_RE.sub(" ", text).strip()
    return text


def _to_refs(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value if v is not None]
    try:
        return [str(v) for v in list(value) if v is not None]
    except Exception:
        return [str(value)]


def _edit_distance(ref: Sequence[str], hyp: Sequence[str]) -> int:
    n = len(ref)
    m = len(hyp)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        r = ref[i - 1]
        for j in range(1, m + 1):
            cost = 0 if r == hyp[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def _wer(ref_text: str, hyp_text: str) -> float:
    ref_words = _normalize(ref_text).split()
    hyp_words = _normalize(hyp_text).split()
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    dist = _edit_distance(ref_words, hyp_words)
    return float(dist) / float(len(ref_words))


def compute_score(prompt_data: Dict[str, Any], response_data: Any):
    """Return reward = max(0, 1 - WER(best_ref, hyp))."""
    is_per_token = False
    correct_threshold = 0.0
    response_ids = list(getattr(response_data, "token_ids", []) or [])
    r = torch.zeros((len(response_ids),), dtype=torch.float32)

    if len(response_ids) == 0:
        return r, is_per_token, correct_threshold

    refs = _to_refs(prompt_data.get("solution", None))
    hyp = str(getattr(response_data, "text", "") or "")
    if not refs:
        r[-1] = 0.0
        return r, is_per_token, correct_threshold

    best_wer = min(_wer(ref, hyp) for ref in refs)
    r[-1] = max(0.0, 1.0 - float(best_wer))
    return r, is_per_token, correct_threshold

