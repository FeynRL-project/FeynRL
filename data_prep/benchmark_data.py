"""
Unified benchmark data preparation for the FeynRL eval loop.

Supported benchmarks:
  aime2024     - AIME 2024         (Maxwell-Jia/AIME_2024)
  aime2025     - AIME 2025 I/II    (opencompass/AIME2025)
  hmmt_feb_25  - HMMT Feb 2025     (MathArena/hmmt_feb_2025)
  hmmt_nov_25  - HMMT Nov 2025     (MathArena/hmmt_nov_2025)
  brumo_2025   - Brumo 2025        (MathArena/brumo_2025)
  amo_bench    - AMO-Bench         (meituan-longcat/AMO-Bench)
  all          - Process all benchmarks above in one run

Usage:
  # AIME 2024
  python benchmark_data.py --benchmark aime2024 --local_dir /path/to/output

  # AIME 2025 — Part I (default) or Part II
  python benchmark_data.py --benchmark aime2025 --local_dir /path/to/output
  python benchmark_data.py --benchmark aime2025 --data_config AIME2025-II --local_dir /path/to/output

  # HMMT February 2025
  python benchmark_data.py --benchmark hmmt_feb_25 --local_dir /path/to/output

  # HMMT November 2025
  python benchmark_data.py --benchmark hmmt_nov_25 --local_dir /path/to/output

  # Brumo 2025
  python benchmark_data.py --benchmark brumo_2025 --local_dir /path/to/output

  # AMO-Bench
  python benchmark_data.py --benchmark amo_bench --local_dir /path/to/output

  # All benchmarks (outputs written to /path/to/data/<benchmark_name>/)
  # aime2025 is processed for both Part I and Part II automatically.
  python benchmark_data.py --benchmark all --local_dir /path/to/data

Common optional flags (all benchmarks):
  --system_prompt "..."   Prepend a system message (omit for no-system-prompt runs)
  --run_id        "..."   Tag embedded in the output filename
  --num_proc      N       Dataset map parallelism (default: 4)
  --data_source   "..."   Override the HuggingFace dataset path (single benchmark only)
  --data_config   "..."   Override the HuggingFace dataset config/subset name (single benchmark only)
"""

import argparse
import os

import datasets


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def create_prompt(question, system_prompt):
    """
    Build a chat-style prompt with an optional system message.
    """
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    return [{"role": "user", "content": question}]


def pick_first_key(example, candidates, field_name):
    """
    Find the first non-null key from the provided candidate list.
    """
    for key in candidates:
        if key in example and example[key] is not None:
            return key

    raise ValueError(
        f"Could not find {field_name} key. Tried: {candidates}. "
        f"Available keys: {list(example.keys())}"
    )


def format_answer(answer):
    """
    Ensure the final answer has the expected separator prefix.
    """
    answer = str(answer).strip()
    if "####" not in answer:
        answer = f"#### {answer}"
    return answer


def normalize_solution(answer):
    """
    Keep compatibility with numeric reward functions used in existing configs.
    """
    return str(answer).replace(",", "").replace("$", "").strip()


def merge_splits_as_test(dataset):
    """
    Flatten every available split into a single test split.
    """
    split_names = list(dataset.keys())
    if not split_names:
        raise ValueError("No splits found in loaded dataset.")

    if len(split_names) == 1:
        return dataset[split_names[0]]

    return datasets.concatenate_datasets([dataset[name] for name in split_names])


# ---------------------------------------------------------------------------
# AIME 2024
# ---------------------------------------------------------------------------


def make_map_fn_aime2024(split, args):
    """
    Convert AIME 2024 rows to the framework format:
    prompt, answer, solution, split, index.
    """

    def process_fn(example, idx):
        question_key = pick_first_key(
            example, ["Problem", "problem", "prompt", "input"], "question"
        )
        solution_key = pick_first_key(
            example, ["Solution", "solution", "explanation"], "solution"
        )
        answer_key = None
        for candidate in ["Answer", "answer", "final_answer", "solution"]:
            if candidate in example and example[candidate] is not None:
                answer_key = candidate
                break

        question = str(example[question_key]).strip()
        solution = str(example[solution_key]).strip()

        if answer_key is None:
            answer_raw = f"#### {solution}"
        else:
            answer_raw = str(example[answer_key]).strip()
            if "####" not in answer_raw:
                answer_raw = f"#### {answer_raw}"

        return {
            "prompt": create_prompt(question, args.system_prompt),
            "answer": answer_raw,
            "solution": solution,
            "split": split,
            "index": idx,
        }

    return process_fn


# ---------------------------------------------------------------------------
# AIME 2025
# ---------------------------------------------------------------------------


def make_map_fn_aime2025(split, args):
    """
    Convert AIME 2025 rows to the framework format:
    prompt, answer, solution, split, index.
    """

    def process_fn(example, idx):
        question_key = pick_first_key(
            example, ["question", "problem", "prompt", "input"], "question"
        )
        solution_key = pick_first_key(
            example, ["solution", "answer", "final_answer", "target"], "solution"
        )
        answer_key = None
        for candidate in ["answer", "solution", "rationale", "explanation"]:
            if candidate in example and example[candidate] is not None:
                answer_key = candidate
                break

        question = str(example[question_key]).strip()
        solution = str(example[solution_key]).strip()

        if answer_key is None:
            answer_raw = f"#### {solution}"
        else:
            answer_raw = str(example[answer_key]).strip()
            if "####" not in answer_raw:
                answer_raw = f"#### {solution}"

        return {
            "prompt": create_prompt(question, args.system_prompt),
            "answer": answer_raw,
            "solution": solution,
            "split": split,
            "index": idx,
        }

    return process_fn


# ---------------------------------------------------------------------------
# HMMT February 2025 and HMMT November 2025 (shared MathArena format)
# ---------------------------------------------------------------------------


def make_map_fn_hmmt(split, args):
    """
    Map MathArena HMMT records to the framework format:
    prompt, answer, solution, split, index.
    Retains problem_idx, subject, round, and name metadata to enable
    filtered eval slices later.
    """

    def process_fn(example, idx):
        problem = example["problem"]
        solution = normalize_solution(example["answer"])
        return {
            "prompt": create_prompt(problem, args.system_prompt),
            "answer": f"#### {solution}",
            "solution": solution,
            "split": split,
            "index": idx,
            # Metadata fields — present on some HMMT subsets only.
            "problem_idx": example.get("problem_idx"),
            "subject": example.get("subject"),
            "round": example.get("round"),
            "name": example.get("name"),
        }

    return process_fn


# ---------------------------------------------------------------------------
# Brumo 2025
# ---------------------------------------------------------------------------


def make_map_fn_brumo(split, args):
    """
    Map MathArena/brumo_2025 records to the framework format:
    prompt, answer, solution, split, index.
    Retains problem_idx and problem_type metadata.
    """

    def process_fn(example, idx):
        problem = example["problem"]
        solution = normalize_solution(example["answer"])
        return {
            "prompt": create_prompt(problem, args.system_prompt),
            "answer": f"#### {solution}",
            "solution": solution,
            "split": split,
            "index": idx,
            "problem_idx": example.get("problem_idx"),
            "problem_type": example.get("problem_type"),
        }

    return process_fn


# ---------------------------------------------------------------------------
# AMO-Bench
# ---------------------------------------------------------------------------


def make_map_fn_amo_bench(split, args):
    """
    Convert AMO-Bench examples into the unified prompt/answer/solution format.
    """

    def process_fn(example, idx):
        question_key = pick_first_key(
            example, ["prompt", "Prompt", "question", "Problem"], "question"
        )
        solution_key = pick_first_key(
            example, ["solution", "Solution"], "solution"
        )

        answer_key = None
        for candidate in ["answer", "Answer", "final_answer"]:
            if candidate in example and example[candidate] is not None:
                answer_key = candidate
                break

        question = str(example[question_key]).strip()
        solution = str(example[solution_key]).strip()
        if answer_key is not None:
            answer_raw = format_answer(example[answer_key])
        else:
            answer_raw = format_answer(solution)

        return {
            "prompt": create_prompt(question, args.system_prompt),
            "answer": answer_raw,
            "solution": solution,
            "answer_type": example.get("answer_type"),
            "question_id": example.get("question_id", idx),
            "split": split,
            "index": idx,
        }

    return process_fn


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

# Each entry specifies:
#   data_source    - default HuggingFace dataset path
#   data_config    - default HuggingFace config/subset (None = no config)
#   default_run_id - tag embedded in output filenames
#   make_map_fn    - row-level processing function factory
#   file_prefix    - leading portion of the output filename
#                    (None = use run_id directly, i.e. HMMT naming convention)
#   hf_split       - specific HF split to load (None = merge all splits)
BENCHMARKS = {
    "aime2024": {
        "data_source": "Maxwell-Jia/AIME_2024",
        "data_config": None,
        "default_run_id": "aime2024_v1",
        "make_map_fn": make_map_fn_aime2024,
        "file_prefix": "aime2024_processed",
        "hf_split": None,
    },
    "aime2025": {
        "data_source": "opencompass/AIME2025",
        "data_config": "AIME2025-I",
        "default_run_id": "aime2025_v1",
        "make_map_fn": make_map_fn_aime2025,
        "file_prefix": "aime2025_processed",
        "hf_split": None,
    },
    "hmmt_feb_25": {
        "data_source": "MathArena/hmmt_feb_2025",
        "data_config": None,
        "default_run_id": "hmmt_feb_2025",
        "make_map_fn": make_map_fn_hmmt,
        "file_prefix": None,  # output: {run_id}_{fpart}_{split}.parquet
        "hf_split": "train",
    },
    "hmmt_nov_25": {
        "data_source": "MathArena/hmmt_nov_2025",
        "data_config": None,
        "default_run_id": "hmmt_nov_2025",
        "make_map_fn": make_map_fn_hmmt,
        "file_prefix": None,  # output: {run_id}_{fpart}_{split}.parquet
        "hf_split": "train",
    },
    "brumo_2025": {
        "data_source": "MathArena/brumo_2025",
        "data_config": None,
        "default_run_id": "brumo_2025",
        "make_map_fn": make_map_fn_brumo,
        "file_prefix": "brumo_2025_processed",
        "hf_split": "train",
    },
    "amo_bench": {
        "data_source": "meituan-longcat/AMO-Bench",
        "data_config": None,
        "default_run_id": "amo_bench_v1",
        "make_map_fn": make_map_fn_amo_bench,
        "file_prefix": "amo_bench_processed",
        "hf_split": None,
    },
}


def create_file_name(run_id, system_prompt, split, config):
    fpart = "wsp" if system_prompt else "ns"
    prefix = config["file_prefix"]
    if prefix is None:
        # HMMT-style: run_id is the full filename prefix.
        return f"{run_id}_{fpart}_{split}.parquet"
    return f"{prefix}_{run_id}_{fpart}_{split}.parquet"


def process_one_benchmark(benchmark_key, config, output_dir, args, data_source=None, data_config=None, run_id=None):
    """
    Load, map, and save a single benchmark shard.

    Parameters
    ----------
    benchmark_key : str
        Key into BENCHMARKS (used only for logging).
    config : dict
        Entry from BENCHMARKS.
    output_dir : str
        Directory where the parquet file will be written.
    args : argparse.Namespace
        Parsed CLI args (system_prompt, num_proc are read from here).
    data_source, data_config, run_id : str | None
        Override values; fall back to config defaults when None.
    """
    effective_source = data_source or config["data_source"]
    effective_config = data_config if data_config is not None else config["data_config"]
    effective_run_id = run_id or config["default_run_id"]

    print(f"\n--- Processing: {benchmark_key} (config={effective_config or 'default'}) ---")

    if effective_config:
        dataset = datasets.load_dataset(effective_source, effective_config)
    else:
        dataset = datasets.load_dataset(effective_source)

    hf_split = config["hf_split"]
    if hf_split is not None:
        test_dataset = dataset[hf_split]
    else:
        test_dataset = merge_splits_as_test(dataset)

    # Pass a lightweight namespace so map functions can access system_prompt.
    map_args = argparse.Namespace(system_prompt=args.system_prompt)
    test_dataset = test_dataset.map(
        function=config["make_map_fn"]("test", map_args),
        with_indices=True,
        num_proc=args.num_proc,
    )

    os.makedirs(output_dir, exist_ok=True)
    fname = create_file_name(effective_run_id, args.system_prompt, "test", config)
    out_path = os.path.join(output_dir, fname)
    test_dataset.to_parquet(out_path)

    print(test_dataset[0])
    print(f"Saved: {out_path}  ({len(test_dataset)} examples)")
    return out_path


# Benchmarks to iterate over in "all" mode.
# aime2025 is listed twice — once for each part.
ALL_SHARDS = [
    ("aime2024",    None),
    ("aime2025",    "AIME2025-I"),
    ("aime2025",    "AIME2025-II"),
    ("hmmt_feb_25", None),
    ("hmmt_nov_25", None),
    ("brumo_2025",  None),
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare benchmark evaluation datasets for the FeynRL eval loop.",
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=list(BENCHMARKS.keys()) + ["all"],
        help=(
            "Which benchmark to process, or 'all' to process every shard "
            "in one run (outputs go to <local_dir>/<benchmark_name>/)."
        ),
    )
    parser.add_argument(
        "--data_source",
        default=None,
        help="HuggingFace dataset path. Defaults to the benchmark's standard source. Ignored for --benchmark all.",
    )
    parser.add_argument(
        "--data_config",
        default=None,
        help="HuggingFace dataset config/subset (e.g. AIME2025-II for aime2025). Ignored for --benchmark all.",
    )
    parser.add_argument("--local_dir", required=True)
    parser.add_argument(
        "--run_id",
        default=None,
        help="Tag embedded in the output filename. Defaults to the benchmark's standard id. Ignored for --benchmark all.",
    )
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant. Think step-by-step and output the final answer after '####'.",
    )
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    if args.benchmark == "all":
        saved = []
        for benchmark_key, shard_config in ALL_SHARDS:
            config = BENCHMARKS[benchmark_key]
            # Use a per-benchmark subdirectory so files don't collide.
            subdir_name = benchmark_key if shard_config is None else f"{benchmark_key}_{shard_config.lower()}"
            output_dir = os.path.join(args.local_dir, subdir_name)
            # For aime2025 parts, suffix the run_id to distinguish the two files.
            run_id = config["default_run_id"]
            if shard_config is not None and shard_config != config.get("data_config"):
                run_id = f"{run_id}_{shard_config.lower()}"
            out_path = process_one_benchmark(
                benchmark_key=benchmark_key,
                config=config,
                output_dir=output_dir,
                args=args,
                data_config=shard_config,
                run_id=run_id,
            )
            saved.append(out_path)

        print("\n\n=== All shards complete ===")
        for p in saved:
            print(f"  {p}")
        print("Done.")

    else:
        config = BENCHMARKS[args.benchmark]

        out_path = process_one_benchmark(
            benchmark_key=args.benchmark,
            config=config,
            output_dir=args.local_dir,
            args=args,
            data_source=args.data_source,
            data_config=args.data_config,
            run_id=args.run_id,
        )

        print(f"\nTest file: {out_path}")
        print("Done.")
