"""RaR-Science preprocessing script.

Loads anisha2102/RaR-Science (pre-split into train/val/test) and converts
each example into the FeynRL parquet format, matching the schema produced by
data_prep/healthbench.py.

Rubric items have {description, title, weight} where weight in {3, 4, 5}.
These map to {criterion, points} for the LLM judge reward function.

Usage:
    python data_prep/rar_science.py \
        --local_dir /path/to/rar_science \
        --run_id v1

Output parquet columns:
    prompt          - list[{role, content}]  (single user turn)
    rubric          - str  (JSON: list[{criterion, points}])
    solution        - str  (reference answer)
    question_source - str
    split           - str
    index           - int
"""
import argparse
import json
import os

import datasets


def _make_map_fn(split: str):
    def process_fn(example, idx):
        rubric_items = []
        for item in (example.get("rubric") or []):
            if not isinstance(item, dict):
                continue
            description = item.get("description")
            weight = item.get("weight")
            if description is None or weight is None:
                continue
            rubric_items.append({"criterion": str(description), "points": int(weight)})

        return {
            "prompt": [{"role": "user", "content": example["question"]}],
            "rubric": json.dumps(rubric_items),
            "solution": str(example.get("reference_answer") or ""),
            "question_source": str(example.get("question_source") or ""),
            "split": split,
            "index": idx,
        }

    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="anisha2102/RaR-Science")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="v1")
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading {args.data_source} ...")
    raw = datasets.load_dataset(args.data_source)
    for split_name in ("train", "val", "test"):
        split_ds = raw[split_name]
        print(f"  {split_name}: {len(split_ds)} rows")

    os.makedirs(args.local_dir, exist_ok=True)

    for split_name in ("train", "val", "test"):
        split_ds = raw[split_name]
        processed = split_ds.map(
            function=_make_map_fn(split_name),
            with_indices=True,
            num_proc=args.num_proc,
            remove_columns=split_ds.column_names,
        )
        out_path = os.path.join(args.local_dir, f"rar_science_{args.run_id}_{split_name}.parquet")
        processed.to_parquet(out_path)
        print(f"{split_name}: {out_path}  ({len(processed)} examples)")

    print("Done.")
