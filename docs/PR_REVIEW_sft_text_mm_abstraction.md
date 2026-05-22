# Code Review: `pr/sft-text-mm-abstraction` (diff vs `main`)

## Executive Summary

This branch is a solid step toward separating **algorithm code** (SFT/PPO/GRPO/…) from **model-family plumbing** (HF loading quirks, forward signatures, device moves, multimodal batch structure). The core idea—introducing a small “model loader registry” plus a “forward adapter” that produces a canonical `(logits, targets, mask)`—is directionally correct and already reduces duplication in `main_sl.py`.

That said, the current implementation is **still mostly “text-only SFT plumbing”**, and the abstraction is only partially wired:

- `models.load_sft_model_and_tokenizer()` is a real improvement and is ready to extend.
- The adapter interface (`ForwardOutput` + `ModelAdapter`) is clean; ensure adapter selection stays aligned with `model_class` as you add modality-specific adapters.
- For future RL work, this abstraction is a good foundation, but **RL needs a richer adapter surface** than “teacher-forcing logits + targets” (e.g., logprobs, generation, value heads, KL/reference evaluation). You’ll want a sibling RL-oriented adapter protocol rather than forcing RL into the SFT `ForwardOutput` shape.

Overall: good foundations, but to “seam” cleanly into image-to-text and RL, it needs a couple of small design pivots (adapter dispatch, and multiple adapter protocols/outputs).

## What Changed (High-Level)

### New model loading dispatch
- Added a lightweight registry in `models/registry.py` with a `@register(name)` decorator and `get_loader(name)`.
- Added `models/load_sft_model_and_tokenizer()` in `models/__init__.py` that dispatches on `model_cfg.model_class`.
- Added HF text loader helpers + registrations in `models/transformers/hf_common.py` and `models/transformers/__init__.py`.

### New forward adapter abstraction for SFT
- Added `models/adapters/base.py`:
  - `ForwardOutput(logits, target_ids, loss_mask)`
  - `ModelAdapter` protocol with `forward()` + `to_device()`
- Added a text-only HF causal LM adapter module (now `models/adapters/hf_causal_lm.py`).
- Added `models/adapters/__init__.py` exporting `get_sft_adapter()`.

### Training loop wiring
- `main_sl.py` now loads the model through `models.load_sft_model_and_tokenizer()` and constructs a `model_adapter` passed into `SFT`.
- `algs/SFT/sft.py` now optionally uses the adapter for forward.
- Added `misc/batch_utils.move_to_device()` and uses it in `main_sl.py` instead of `v.to(device)` (better for nested/multimodal batches).

### Misc improvements
- `misc/checkpoint_utils.py` adds a pragmatic mitigation for ZeRO-3 “param in flight” assertions before gathering weights for save.
- `data_feeds/mixed_sampler.py` adds stronger validation around dataset keys/empties and makes RNG seeding more robust.
- Added offline-friendly synthetic dataset generator `data_prep/synthetic.py`.
- Added multiple unit/integration tests covering loader registry, adapters, and offline SFT smoke.

## Code Review Notes by Area

### 1) `models/registry.py` (loader registry)

**Strengths**
- Minimal, readable API. Decorator-based registration is ergonomic.
- `get_loader()` errors include “Available: …” which is helpful.

**Concerns / suggestions**
- `list_loaders()` returns insertion-order list; callers typically want `sorted(list_loaders())` for stable logs/errors.
- Consider typing the registry more strongly (e.g., “callable returning `(model, tokenizer)`”), even if kept as `Callable[..., Tuple[Any, Any]]`.
- Side-effect registration relies on importing family modules (`import models.transformers`). This is fine, but it should be made explicit in documentation and/or enforced in a single `models/__init__.py` import point (which you did).

### 2) `models/transformers/hf_common.py` (HF text loader)

**Strengths**
- Extracting “pad token normalization” and “HF load with dtype + attn_implementation” from `main_sl.py` is a net win.
- The guardrails are good:
  - `dtype != "auto"` to avoid silent precision surprises.
  - validated `attn_implementation`.

**Concerns / suggestions**
- `@register("llm")` is broad/ambiguous and may conflict with future non-HF or multimodal classes. Consider namespacing loader keys (`transformers_llm_text`, `transformers_{family}_{task}`) to avoid collisions.
- For multimodal models, you’ll likely need:
  - a different base loader (e.g., `AutoProcessor` / `AutoModelForVision2Seq`) or processor + tokenizer combos
  - special handling for `pad_token_id`, image token ids, etc.
  - trust_remote_code quirks
  This file is a good place to add “family common” helpers, but keep the registrations granular.

### 3) `models/adapters/base.py` and `models/adapters/qwen_2.py`

**Strengths**
- `ForwardOutput` is a clear canonical SFT interface.
- `ModelAdapter.to_device()` is a strong design choice for multimodal: it allows adapters to own device-move semantics.
- `misc/batch_utils.move_to_device()` is robust enough for nested dict/list/tuple and HF-like BatchFeatures.

**Concerns / suggestions**
- Keep adapter modules model-/family-specific as needed (verl/AReaL style), but name “generic” adapters generically to avoid confusion.
- The adapter currently assumes batch keys are exactly:
  - `input_ids`, `attn_mask`, `loss_mask`, optional `position_ids`
  That’s consistent with `PairedFeed`, but for multimodal you’ll need additional keys like `pixel_values` / `images` / `image_grid_thw` etc. This will be fine if the adapter owns the batch interpretation (it should).

### 4) `models/adapters/__init__.py` (`get_sft_adapter`)

This is the largest “abstraction seam” issue right now:

- Adapter selection must not silently return the wrong adapter once multimodal/model-specific adapters are introduced.

**Why it matters**
- The branch name and intent suggest “text mm abstraction”. The moment you add a real multimodal adapter, you’ll accidentally keep using the text adapter unless you remember to update this function everywhere.
- It also defeats the point of having `model_cfg.model_class`—you already have the exact dispatch key needed to select an adapter.

**Recommendation**
- Make adapter selection symmetrical with loader selection:
  - either add an `adapters` registry keyed by `model_class`, or
  - have the loader return `(model, tokenizer, adapter)` (or a small `ModelBundle` object) so loading and adapter selection can’t drift.

### 5) `main_sl.py` wiring + `algs/SFT/sft.py`

**Strengths**
- Moving model loading out of `main_sl.py` removes a lot of duplicated boilerplate.
- Passing `model_adapter` into `SFT` is a clean dependency injection point.
- Replacing `{k: v.to(device)}` with `move_to_device()` is a good preparatory step for multimodal batches.

**Concerns / suggestions**
- `main_sl.py` imports `move_to_device` directly and uses it, but the abstraction has `ModelAdapter.to_device()`. For future multimodal, you’ll likely want the training loop to call `model_adapter.to_device(batch, device)` instead of `move_to_device(batch, device)` so model-family specifics live in one place.
- SFT’s adapter hook only affects the forward pass. That’s fine for now, but consider whether SFT should fully own “batch → loss inputs” via the adapter (including masks and any shifting). You’re mostly there.

### 6) `misc/checkpoint_utils.py` ZeRO-3 mitigation

**Strengths**
- The comment explains the “in flight partition” failure mode clearly.
- The mitigation is appropriately best-effort (warnings, not hard failure).

**Concerns / suggestions**
- If this becomes a recurring problem, consider making “checkpoint stabilization” a named helper, and/or gating it behind config flags so it’s easy to bisect performance/behavior.

### 7) `data_feeds/mixed_sampler.py`

**Strengths**
- Stronger input validation (dataset key mismatch, empty datasets) will prevent subtle “silent sampling bugs”.
- Better per-rank/per-epoch RNG seeding via `SeedSequence` reduces correlations.

**Concerns / suggestions**
- Not directly related to the model abstraction; fine as drive-by correctness improvements, but keep an eye on PR scope creep long-term.

### 8) Tests (`unit_tests/...`)

**Strengths**
- Good balance:
  - unit tests for the registry/loader validation
  - integration-style smoke tests that avoid network downloads by using `TinyModel` and mock tokenizers
- The `pytest.importorskip("data_feeds.paired", reason=...)` pattern is pragmatic in environments where optional deps might be missing.

**Concerns / suggestions**
- The integration tests assume `PairedFeed` exists and behaves consistently; if `PairedFeed` changes, these tests may become brittle. Consider adding a smaller “pure tensor batch” test that bypasses dataset dependencies.

## How Well Does This “Seam” With an Image-to-Text Example?

Pretty well *in principle*, with two key caveats:

1) **You need adapter dispatch.**
   If `get_sft_adapter()` always returns the text adapter, it will block true multimodal support in practice.

2) **Device-move should be adapter-owned.**
   Multimodal batches often contain:
   - tensors that must move (`pixel_values`, `input_ids`, attention masks)
   - non-tensors that should not (`images` as PIL objects, metadata)
   - nested processor outputs (HF BatchFeature)
   You already created `ModelAdapter.to_device()` and a robust `move_to_device()`—the missing step is routing *all* batch device logic through the adapter.

In other words: the abstraction is close to being a clean foundation for `pr/text-to-image-example`, but it needs the “dispatch wiring” finished so the correct adapter is selected and owns device semantics.

## How Well Does This “Seam” With RL (PPO/GRPO/… in `algs/RL/common.py`)?

This abstraction is a good *starting point*, but RL will require a broader interface than SFT’s `ForwardOutput`.

### Where it aligns
- RL code in `algs/RL/common.py` already implements a “policy forward” that:
  - calls `engine(input_ids, attention_mask, position_ids, use_cache=False)`
  - shifts logits to `[B, T-1, V]`
  - computes per-token logprobs via a fused helper
  This is conceptually the same “model-family plumbing” you’re extracting for SFT.

### Where it doesn’t yet fit
RL algorithms need more than `(logits, targets, loss_mask)`:
- **logprobs** for arbitrary actions (not only “next token of input_ids”)
- **generation** (sampling, temperature/top-p) and consistency with tokenizer/processor
- **reference model forward** (KL) and possibly reward model forward
- optional **value head** outputs (PPO)
- careful handling of **masks/dones** that differ from SFT’s `loss_mask`

### Recommendation for RL integration
Instead of stretching `ForwardOutput`, define a sibling protocol/output for RL:

- `RLPolicyAdapter` (or `PolicyAdapter`) methods like:
  - `forward_logits(model_engine, batch) -> logits` (or a richer output)
  - `logprobs_from_logits(logits, target_ids, mask) -> logprobs`
  - `generate(model_engine, batch, gen_cfg) -> sequences + masks`
  - `to_device(batch, device) -> batch`

Then refactor `algs/RL/common.py` to use:
- the same `models` loader registry (so SL/RL share model load code),
- RL adapters for forward/logprobs/generation,
while keeping algorithm math (PPO/GRPO losses) in algorithm classes.

This would remove a lot of duplicated HF/DeepSpeed plumbing in RL and make it much easier to extend to multimodal policies later.

## Small Action Items to Make the Abstraction “Feel Finished”

If you want this PR to be a stable base for `pr/text-to-image-example` and future RL work, the highest-leverage follow-ups are:

1) **Dispatch adapters based on `model_class`** (or return adapter from the loader).
2) **Have `main_sl.py` call `model_adapter.to_device()`** (not the global `move_to_device()`), so multimodal device semantics are centralized.
3) **Keep adapter names honest** (generic vs model-specific) to reduce future confusion.
4) **Introduce an RL-specific adapter/output** rather than forcing RL through `ForwardOutput`.
5) **Optional:** Namespace/standardize loader keys (`transformers_*`) to avoid future key collisions.

## Minor Nits / Hygiene

- `.gitignore` currently contains `configs/murdock_configs/` twice (it existed and is re-added). It’s harmless but slightly noisy.
- Consider stabilizing `list_loaders()` output ordering in user-facing error messages/logs.
