import os
import sys
import types
import importlib.util
from unittest.mock import MagicMock

# data_feeds.preference does `from datasets import load_dataset` at import time. The test
# harness mocks huggingface_hub, which breaks the real `datasets` import. We never call
# load_dataset here (we bypass __init__/_load_data), so a lightweight stub is enough.
if "load_dataset" not in dir(sys.modules.get("datasets", object())):
    _ds = types.ModuleType("datasets")
    _ds.load_dataset = MagicMock()
    sys.modules["datasets"] = _ds

import torch

# Load preference.py directly by file path. Another test stubs `data_feeds` as a non-package
# in sys.modules, so `from data_feeds.preference import ...` is unreliable in the full suite;
# preference.py has no intra-package imports, so a standalone load is clean and isolated.
_PREF_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data_feeds", "preference.py")
_spec = importlib.util.spec_from_file_location("preference_standalone", _PREF_PATH)
_pref_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pref_mod)
PreferenceFeed = _pref_mod.PreferenceFeed


class _FakeTok:
    '''Minimal tokenizer for _process_answer/_check_seq (no chat template needed since
    _encode_prompt is overridden in the mm test).'''
    eos_token_id = 99
    pad_token_id = 0

    def __call__(self, text, return_tensors='pt', add_special_tokens=False):
        ids = torch.tensor([[1, 2, 3]])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _make(cls, max_seq_len=32):
    # Bypass __init__ (which loads a parquet) and set only what _get_sample needs.
    feed = object.__new__(cls)
    feed.tokenizer = _FakeTok()
    feed.max_seq_len = max_seq_len
    return feed


def test_base_get_sample_invokes_mm_hooks_and_merges():
    '''
        Regression guard: the base PreferenceFeed._get_sample MUST call _encode_prompt and
        _pair_mm and merge the multimodal tensors into the output. If it tokenizes directly
        (the old bug), a VLM subclass's image tensors would be silently dropped.
    '''
    calls = {"encode": 0, "pair": 0}

    class _MMPref(PreferenceFeed):
        def _encode_prompt(self, message, add_generation_prompt=True):
            calls["encode"] += 1
            # pretend the processor expanded an image into 4 prompt tokens
            return torch.tensor([5, 6, 7, 8], dtype=torch.long), {"pixel_values": torch.zeros(2, 3)}

        def _pair_mm(self, mm, prompt_len, seq_len):
            calls["pair"] += 1
            assert prompt_len == 4 and seq_len == self.max_seq_len
            # duplicate the bag for chosen + rejected (as ImagePreferenceFeed does)
            return {"pixel_values": torch.cat([mm["pixel_values"], mm["pixel_values"]], dim=0)}

    feed = _make(_MMPref)
    out = feed._get_sample(idx=0, message=[{"role": "user", "content": "x"}],
                           chosen_answer="good", rejected_answer="bad")

    assert calls["encode"] == 1 and calls["pair"] == 1
    # mm tensor is merged into the per-sample dict (duplicated -> [4, 3])
    assert "pixel_values" in out and out["pixel_values"].shape == (4, 3)
    # paired text tensors keep their [2, T] / [2, T-1] shapes
    assert out["input_ids"].shape[0] == 2 and out["input_ids"].shape[1] == feed.max_seq_len
    assert out["loss_mask"].shape == (2, feed.max_seq_len - 1)


def test_base_text_only_returns_no_mm_keys():
    '''
        LLM path parity: with the default _encode_prompt/_pair_mm (overridden here only to
        avoid needing a real chat template), the output carries no multimodal keys.
    '''
    class _TextPref(PreferenceFeed):
        def _encode_prompt(self, message, add_generation_prompt=True):
            return torch.tensor([5, 6, 7, 8], dtype=torch.long), {}
        # _pair_mm inherited from base -> returns {}

    feed = _make(_TextPref)
    out = feed._get_sample(idx=0, message=[{"role": "user", "content": "x"}],
                           chosen_answer="good", rejected_answer="bad")

    assert set(out.keys()) == {"input_ids", "attn_mask", "loss_mask"}
