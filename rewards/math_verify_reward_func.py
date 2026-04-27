import logging
import multiprocessing
from typing import Callable, Optional, Sequence, Dict, Any
import torch
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from math_verify.errors import TimeoutException
from math_verify.grader import verify
from math_verify.parser import ExprExtractionConfig, ExtractionTarget, LatexExtractionConfig, parse

# Adapted from https://github.com/verl-project/verl/blob/d9d94b4da93fbacc06bb546629171c67c0a674aa/verl/utils/reward_score/math_reward.py

logger = logging.getLogger(__name__)

# Lazy-initialized process pool for reward computation.
# Subprocesses have their own main thread, so signal.SIGALRM (used by
# math_verify's @timeout decorator) works correctly — unlike in the
# async/overlap engine where the Ray actor may not be on the main thread.
#
# IMPORTANT: We use "spawn" (not the default "fork") because this pool
# is created inside a Ray actor that already has an active CUDA context,
# NCCL communicators, and torch.distributed state.  Plain fork() would
# duplicate that state into the child, causing silent corruption or hangs.
# "spawn" starts a fresh Python interpreter via exec — no inherited state.
# Slightly slower first-time startup (children re-import modules), but the
# pool is created once and workers are reused across all subsequent calls.
# _run_verification is pure-CPU (sympy/math_verify) and needs no GPU.
_REWARD_POOL: Optional[ProcessPoolExecutor] = None
_REWARD_POOL_WORKERS = 8
_MP_CTX = multiprocessing.get_context("spawn")

def _get_reward_pool() -> ProcessPoolExecutor:
    global _REWARD_POOL
    if _REWARD_POOL is None:
        _REWARD_POOL = ProcessPoolExecutor(max_workers=_REWARD_POOL_WORKERS,
                                           mp_context=_MP_CTX)
    return _REWARD_POOL

def math_metric(gold_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
                pred_extraction_target: Sequence[ExtractionTarget] = (ExprExtractionConfig(),),
                aggregation_function: Callable[[list[float]], float] = max,
                precision: int = 6) -> Callable[[list[str], list[str]], tuple[float, Optional[tuple[list[str], list[str]]]]]:
    '''
        Creates a language-aware extractive match metric that extracts answers from the model's output.
        Known issues:
        - If the task is to simplify an expression, the metric might overestimate the accuracy. This is because if the model doesn't output any anchor for the extraction (e.g final answer is..),
            it's possible that the the extracted prediction will be the expression to simplify. Because we do simplifications ourselves, it can thus happen that sympy will correctly simplify the expression,
            thus it will match gold, despite model not doing anything. PRs to fix this are welcome.

        Args:
            gold_extraction_target: Sequence[ExtractionTarget]
                Extraction targets to use for gold answers. Defaults to extracting simple math expressions.
            pred_extraction_target: Sequence[ExtractionTarget]
                Extraction targets to use for predictions. Defaults to extracting simple math expressions.
            aggregation_function: Callable[[list[float]], float]
                Function to aggregate scores when multiple golds/predictions are present. Defaults to max.
            precision: int
                Number of decimal places to use when comparing numerical values. Defaults to 6.

        Returns:
            A sample level metric that extracts and compares mathematical expressions.
    '''

    def get_str_preds_with_timeout(extracted_predictions: list[list[str]], extracted_golds: list[list[str]]) -> tuple[list[str], list[str]]:
        golds = [str(gold) for golds in extracted_golds for gold in golds]
        predictions = [str(pred) for preds in extracted_predictions for pred in preds]
        return (golds, predictions)

    def sample_level_fn(golds: list[str], predictions: list[str]) -> tuple[float, Optional[tuple[list[str], list[str]]]]:
        extracted_predictions = [parse(pred, pred_extraction_target, parsing_timeout=None) for pred in predictions]
        extracted_golds = [parse(gold, gold_extraction_target, parsing_timeout=None) for gold in golds]

        # Assert on empty gold and warn on empty pred
        if any(len(g) == 0 for g in extracted_golds):
            raise ValueError(f"No gold targets found for at least one gold. Gold: {golds}, Pred: {predictions}")

        # We have to use timeout because the sypmy to str conversion can be very slow
        str_preds = None
        try:
            str_preds = get_str_preds_with_timeout(extracted_predictions, extracted_golds)
        except TimeoutException:
            logger.warning("Timeout when adding extracted predictions and golds to specific")

        return (
            aggregation_function([(1.0 if any(verify(gold, pred, precision, timeout_seconds=None) for gold in extracted_golds) else 0.0) for pred in extracted_predictions]),
            str_preds,
        )

    return sample_level_fn


def _run_verification(ground_truth_boxed: str, solution_str: str, timeout_score: float) -> float:
    '''
        Top-level function executed in a subprocess via ProcessPoolExecutor.
        signal.SIGALRM (used by math_verify's @timeout decorator) works here
        because each subprocess has its own main thread.

        LatexExtractionConfig and ExprExtractionConfig are run as separate parse
        passes, each with its own independent timeout. Previously they shared a
        single timeout: ExprExtractionConfig scanning a long chain-of-thought
        would exhaust the budget before LatexExtractionConfig could find \boxed{},
        causing correct answers to receive 0 reward.
    '''
    # Suppress math_verify's "Timeout during parsing: <full model text>" warning,
    # which would otherwise flood stderr with model completions via Ray actor logs.
    logging.getLogger("math_verify").setLevel(logging.ERROR)

    _latex_verify = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                                pred_extraction_target=(LatexExtractionConfig(),))
    _expr_verify  = math_metric(gold_extraction_target=(LatexExtractionConfig(),),
                                pred_extraction_target=(ExprExtractionConfig(),))

    best = 0.0
    for verify_func in (_latex_verify, _expr_verify):
        try:
            score, _ = verify_func([ground_truth_boxed], [solution_str])
            best = max(best, float(score))
            if best == 1.0:
                return best  # can't do better; skip remaining passes
        except TimeoutException:
            best = max(best, float(timeout_score))
        except Exception as e:
            logging.getLogger(__name__).error(f"_run_verification failed: {e}")
    return best


def compute_scores_batch(pairs: list[tuple[Dict[str, Any], Any]],
                         timeout_score: float = 0.0,
                         per_call_timeout: float = 60.0) -> list[tuple[torch.Tensor, bool, float]]:
    '''
        Score all (prompt_data, response_data) pairs concurrently via the
        process pool.  All futures are submitted first (non-blocking), then
        collected — so up to _REWARD_POOL_WORKERS verifications run in parallel.

        Returns a list with the same length and order as `pairs`, where each
        element is (rewards_tensor, is_per_token, correct_threshold).
    '''
    pool = _get_reward_pool()
    is_per_token = False
    correct_threshold = 0.0

    # Phase 1: submit all jobs without blocking
    submissions = []  # (future | None, n_tokens)
    for prompt_data, response_data in pairs:
        n_tokens = len(response_data.token_ids)
        if n_tokens == 0:
            submissions.append((None, n_tokens))
            continue
        ground_truth = prompt_data["solution"]
        ground_truth_boxed = ground_truth
        if "\\boxed" not in ground_truth:
            ground_truth_boxed = "\\boxed{" + ground_truth + "}"
        future = pool.submit(_run_verification, ground_truth_boxed,
                             response_data.text, timeout_score)
        submissions.append((future, n_tokens))

    # Phase 2: collect all results (workers run concurrently in between)
    results = []
    for future, n_tokens in submissions:
        r = torch.zeros((n_tokens,), dtype=torch.float32, device='cpu')
        if future is not None:
            try:
                r[-1] = future.result(timeout=per_call_timeout)
            except FuturesTimeoutError:
                future.cancel()  # only prevents queued tasks; cannot kill running worker
                logger.warning(f"Reward computation exceeded {per_call_timeout}s wall-clock cap "
                               f"(internal signal.alarm should have fired at 30s). "
                               f"Returning timeout_score={timeout_score}.")
                r[-1] = timeout_score
            except Exception as e:
                logger.error(f"Error in math_verify compute_scores_batch: {e}")
                r[-1] = 0.0
        results.append((r, is_per_token, correct_threshold))
    return results


def compute_score(prompt_data: Dict[str, Any], response_data: Dict[str, Any],
                  timeout_score: float = 0.0, per_call_timeout: float = 60.0):
    '''
      Single-pair scoring. Delegates to compute_scores_batch for consistency.

      input args:
        prompt_data: Dict[str, Any]
        response_data: Dict[str, Any]
        timeout_score: score to return on timeout
        per_call_timeout: hard wall-clock cap (seconds) for the entire
            verification. Prevents stacking of internal 30s timeouts
            (parse + str-conversion + verify) from exceeding a safe bound.
      output args:
        r: torch.Tensor of length of response token ids
        is_per_token: whether the reward is per token
        correct_threshold: a response is counted as correct for pass@k
            when its scalar reward strictly exceeds this threshold. For example, for binary
            reward functions [0,1], this threshold is 0.0.
    '''
    return compute_scores_batch([(prompt_data, response_data)],
                                timeout_score=timeout_score,
                                per_call_timeout=per_call_timeout)[0]


# Attach batch function so the engine can discover it via
# hasattr(reward_func, 'batch') without changing the constructor interface.
compute_score.batch = compute_scores_batch