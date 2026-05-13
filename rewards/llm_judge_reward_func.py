import json
import logging
import re
from typing import Any, Dict, Optional

import torch
from openai import APIConnectionError, APITimeoutError, OpenAI

logger = logging.getLogger(__name__)

_RETRYABLE = (APITimeoutError, APIConnectionError)


def _coerce_bool(val: Any) -> Optional[bool]:
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ("true", "false"):
            return v == "true"
    return None


def _parse_criteria_met(text: str) -> Optional[bool]:
    if not text:
        return None
    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            data = json.loads(c)
            if "criteria met" in data:
                coerced = _coerce_bool(data["criteria met"])
                if coerced is not None:
                    return coerced
        except Exception:
            continue
    m = re.search(r'"criteria met"\s*:\s*(true|false)', text, re.IGNORECASE)
    return m.group(1).lower() == "true" if m else None


class _LLMJudgeScorer:
    def __init__(self):
        self._cfg = {}
        self._client = None

    def configure(self, reward_config) -> None:
        system_prompt_path = getattr(reward_config, "judge_system_prompt_path", None)
        if not system_prompt_path:
            raise RuntimeError("[LLMJudge] reward.judge_system_prompt_path is required.")
        with open(system_prompt_path) as f:
            system_prompt = f.read().strip()

        base_url = getattr(reward_config, "judge_base_url", None) or ""
        model_id = getattr(reward_config, "judge_model", None) or ""
        if not base_url:
            raise RuntimeError("[LLMJudge] reward.judge_base_url is required.")
        if not model_id:
            raise RuntimeError("[LLMJudge] reward.judge_model is required.")

        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"

        timeout_s = float(getattr(reward_config, "judge_timeout", 60.0) or 60.0)
        max_retries = int(getattr(reward_config, "judge_max_retries", 2) or 2)
        max_tokens = int(getattr(reward_config, "judge_max_tokens", 4096) or 4096)
        enable_thinking = getattr(reward_config, "judge_enable_thinking", None)
        extra_body = (
            {"chat_template_kwargs": {"enable_thinking": bool(enable_thinking)}}
            if enable_thinking is not None
            else None
        )

        logger.info(f"[LLMJudge] model={model_id!r}, endpoint={base_url!r}, prompt={system_prompt_path!r}")
        self._cfg = dict(
            base_url=base_url,
            model_id=model_id,
            system_prompt=system_prompt,
            timeout_s=timeout_s,
            max_retries=max_retries,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        self._client = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_client"] = None  # OpenAI client is not picklable; re-created on first call
        return state

    def __call__(self, prompt_data: Dict[str, Any], response_data: Any):
        if not self._cfg:
            raise RuntimeError("[LLMJudge] configure() must be called before compute_score().")

        response_ids = list(response_data.token_ids)
        r = torch.zeros(len(response_ids), dtype=torch.float32)
        if not response_ids:
            return r, False, 0.0

        rubric_val = prompt_data.get("rubric", "")
        if not rubric_val:
            return r, False, 0.0

        if isinstance(rubric_val, str):
            try:
                rubric_val = json.loads(rubric_val)
            except Exception:
                logger.warning("[LLMJudge] Failed to parse rubric JSON; assigning score=0.0")
                return r, False, 0.0

        if not isinstance(rubric_val, list) or not rubric_val:
            return r, False, 0.0

        rubric_items = [
            {"criterion": str(it["criterion"]), "points": int(it["points"])}
            for it in rubric_val
            if isinstance(it, dict) and "criterion" in it and "points" in it
        ]
        if not rubric_items:
            return r, False, 0.0

        max_pos = sum(max(0, it["points"]) for it in rubric_items)
        min_neg = sum(min(0, it["points"]) for it in rubric_items)
        if max_pos == min_neg:
            r[-1] = 0.5
            return r, False, 0.0

        if self._client is None:
            self._client = OpenAI(
                base_url=self._cfg["base_url"],
                api_key="dummy",
                timeout=self._cfg["timeout_s"],
            )

        convo = f"{prompt_data.get('text', '')}{response_data.text}"
        total_points = 0
        for it in rubric_items:
            user_msg = (
                f"# Conversation\n{convo}\n\n"
                f"# Rubric item\n{json.dumps({'criterion': it['criterion']}, ensure_ascii=False)}"
            )
            met = None
            for attempt in range(1, self._cfg["max_retries"] + 1):
                try:
                    kwargs = {"extra_body": self._cfg["extra_body"]} if self._cfg["extra_body"] else {}
                    resp = self._client.chat.completions.create(
                        model=self._cfg["model_id"],
                        messages=[
                            {"role": "system", "content": self._cfg["system_prompt"]},
                            {"role": "user", "content": user_msg},
                        ],
                        max_tokens=self._cfg["max_tokens"],
                        temperature=0.0,
                        response_format={"type": "json_object"},
                        **kwargs,
                    )
                    msg = resp.choices[0].message
                    text = (
                        getattr(msg, "content", None)
                        or getattr(msg, "reasoning_content", None)
                        or getattr(msg, "reasoning", None)
                        or ""
                    )
                    met = _parse_criteria_met(text)
                    break
                except _RETRYABLE as exc:
                    if attempt >= self._cfg["max_retries"]:
                        logger.warning(f"[LLMJudge] retryable error after {self._cfg['max_retries']} attempts: {exc}")
                except Exception as exc:
                    logger.warning(f"[LLMJudge] judge call error: {type(exc).__name__}: {exc}")
                    break

            if met:
                total_points += it["points"]

        normalized = (total_points - min_neg) / float(max_pos - min_neg)
        r[-1] = float(max(0.0, min(1.0, normalized)))
        return r, False, 0.0


compute_score = _LLMJudgeScorer()
configure = compute_score.configure
