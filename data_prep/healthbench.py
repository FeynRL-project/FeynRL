"""HealthBench preprocessing script.

Loads openai/healthbench, deduplicates by prompt_id (keeping one example per
unique prompt), stores the per-example rubric as a JSON string, and saves
train/val parquets under --local_dir.

Usage:
    python data_prep/healthbench.py \
        --local_dir /path/to/healthbench \
        --val_ratio 0.1 \
        --seed 42

Output parquet columns:
    prompt      - list[{role, content}]  (tokenizable chat messages)
    rubric      - str  (JSON: list[{criterion, points, tags}])
    prompt_id   - str
    category    - str
    example_tags - list[str]
    split       - str
    index       - int
"""
import argparse
import json
import os

import datasets
from datasets import Features, Value


# Minimal feature spec covering the fields we actually use.
_HB_FEATURES = Features({
    "prompt": [{"content": Value("string"), "role": Value("string")}],
    "prompt_id": Value("string"),
    "rubrics": [{"criterion": Value("string"), "points": Value("int64"), "tags": [Value("string")]}],
    "canary": Value("string"),
    "example_tags": [Value("string")],
    "ideal_completions_data": {
        "ideal_completion": Value("string"),
        "ideal_completions_group": Value("string"),
        "ideal_completions_ref_completions": [Value("string")],
    },
    "anonymized_physician_ids": [Value("string")],
    "binary_labels": [Value("bool")],
    "category": Value("string"),
    "completion": Value("string"),
    "completion_id": Value("string"),
    "rubric": Value("string"),
})


def _make_map_fn(split: str, params):
    def process_fn(example, idx):
        # Normalise rubric list — drop entries where points is None.
        raw_rubrics = example.get("rubrics") or []
        rubrics = [
            {"criterion": r["criterion"], "points": int(r["points"]), "tags": list(r["tags"] or [])}
            for r in raw_rubrics
            if r.get("points") is not None
        ]

        return {
            "prompt": example["prompt"],
            "rubric": json.dumps(rubrics),
            "prompt_id": str(example.get("prompt_id") or f"{split}_{idx}"),
            "category": str(example.get("category") or ""),
            "example_tags": list(example.get("example_tags") or []),
            "split": split,
            "index": idx,
        }

    return process_fn


def _dedup_by_prompt_id(dataset):
    seen = set()
    keep = []
    for i, pid in enumerate(dataset["prompt_id"]):
        if pid not in seen:
            seen.add(pid)
            keep.append(i)
    return dataset.select(keep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="openai/healthbench")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="v1")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.data_source} ...")
    raw = datasets.load_dataset(args.data_source, split="test", features=_HB_FEATURES)
    print(f"Raw dataset: {len(raw)} rows")

    # HealthBench test split contains multiple scored completions per prompt.
    # Deduplicate so each unique prompt appears exactly once.
    unique = _dedup_by_prompt_id(raw)
    print(f"After dedup by prompt_id: {len(unique)} unique prompts")

    split = unique.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_raw = split["train"]
    val_raw = split["test"]

    train_ds = train_raw.map(
        function=_make_map_fn("train", args),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=train_raw.column_names,
    )
    val_ds = val_raw.map(
        function=_make_map_fn("val", args),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=val_raw.column_names,
    )

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, f"healthbench_{args.run_id}_train.parquet")
    val_path = os.path.join(args.local_dir, f"healthbench_{args.run_id}_val.parquet")

    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)

    print(f"\nTrain: {train_path}  ({len(train_ds)} examples)")
    print(f"Val:   {val_path}  ({len(val_ds)} examples)")
    print("Done.")
