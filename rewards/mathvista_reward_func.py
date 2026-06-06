import json
import re
import torch
from typing import Any, Dict


def _extract_option_letter(text: str) -> str | None:
    """Return the last isolated A/B/C/D letter in the response."""
    hits = re.findall(r'(?<![A-Za-z])([A-D])(?![A-Za-z])', text)
    return hits[-1].upper() if hits else None


def _extract_last_number(text: str) -> float | None:
    """Return the last integer or decimal number appearing in the response."""
    hits = re.findall(r'-?\d+(?:\.\d+)?', text)
    if not hits:
        return None
    try:
        return float(hits[-1])
    except ValueError:
        return None


def _numbers_match(pred: float, gold_str: str, precision: float) -> bool:
    try:
        gold = float(gold_str)
    except (ValueError, TypeError):
        return False
    # tolerance = half a unit at the given decimal precision
    tol = 0.5 * (10.0 ** (-max(0, int(precision))))
    return abs(pred - gold) <= tol


def compute_score(prompt_data: Dict[str, Any], response_data: Any,
                  timeout_score: float = 0.0, per_call_timeout: float = 60.0):
    """
    Score a MathVista response.

    prompt_data["solution"] must be a JSON string produced by data_prep/mathvista.py:
      - multi_choice: {"answer": "C", "question_type": "multi_choice"}
      - free_form:    {"answer": "1.2", "question_type": "free_form",
                       "answer_type": "float", "precision": 1.0}

    Returns:
        r                : torch.Tensor of length n_response_tokens (reward at last token)
        is_per_token     : False  (scalar reward)
        correct_threshold: 0.0   (reward > 0 counts as correct for pass@k)
    """
    n_tokens = len(response_data.token_ids)
    r = torch.zeros(n_tokens, dtype=torch.float32)
    is_per_token = False
    correct_threshold = 0.0

    if n_tokens == 0:
        return r, is_per_token, correct_threshold

    try:
        meta = json.loads(prompt_data["solution"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return r, is_per_token, correct_threshold

    answer = str(meta.get("answer", "")).strip()
    qt = meta.get("question_type", "free_form")
    at = meta.get("answer_type", "text")
    precision = float(meta.get("precision", 1.0))
    response_text = response_data.text

    score = 0.0
    if qt == "multi_choice":
        extracted = _extract_option_letter(response_text)
        if extracted is not None:
            score = 1.0 if extracted == answer.upper() else 0.0
    else:
        if at in ("integer", "float"):
            extracted_num = _extract_last_number(response_text)
            if extracted_num is not None:
                score = 1.0 if _numbers_match(extracted_num, answer, precision) else 0.0
        else:
            # text / list: case-insensitive substring match
            score = 1.0 if answer.lower() in response_text.lower() else 0.0

    r[-1] = score
    return r, is_per_token, correct_threshold
