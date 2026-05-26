from __future__ import annotations

import numpy as np
import pandas as pd


class _TokenizerKwBreaks:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, *args, **kwargs):
        if "conversation" in kwargs:
            raise TypeError("can only concatenate str (not \"list\") to str")
        if kwargs.get("tokenize", False):
            return [1, 2, 3]
        return "PROMPT"


class _Adapter:
    def prepare_messages(self, messages):
        out = []
        injected = False
        for turn in messages:
            if (not injected) and turn.get("role") == "user" and isinstance(turn.get("content"), str):
                out.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": "placeholder"},
                            {"type": "text", "text": turn["content"]},
                        ],
                    }
                )
                injected = True
            else:
                if isinstance(turn.get("content"), str):
                    out.append({"role": turn.get("role"), "content": [{"type": "text", "text": turn["content"]}]})
                else:
                    out.append(turn)
        return out


def test_audio_prompts_feed_falls_back_to_processor_signature(tmp_path):
    from data_feeds.audio_prompts import AudioPromptsFeed

    df = pd.DataFrame(
        [
            {
                "prompt": [{"role": "user", "content": "transcribe"}],
                "answers": ["HELLO"],
                "audio_bytes": np.zeros((160,), dtype=np.float32),
                "sampling_rate": 16000,
            }
        ]
    )
    p = tmp_path / "asr.parquet"
    df.to_parquet(p, index=False)

    ds = AudioPromptsFeed(
        prompt_key="prompt",
        tokenizer=_TokenizerKwBreaks(),
        max_seq_len=256,
        data_path=str(p),
        solution_key="answers",
        adapter=_Adapter(),
    )

    item = ds[0]
    assert item["prompt"] == "PROMPT"
    assert "multi_modal_data" in item

