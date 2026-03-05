import argparse
import os
import datasets


def create_prompt(question, system_prompt):
    """
    Create chat-formatted prompt with optional system message.
    """
    if system_prompt:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

    return [{"role": "user", "content": question}]


def pick_first_key(example, candidates, field_name):
    """
    Return the first existing key from candidates, or raise a clear error.
    """
    for key in candidates:
        if key in example and example[key] is not None:
            return key
    raise ValueError(
        f"Could not find {field_name} key. Tried: {candidates}. "
        f"Available keys: {list(example.keys())}"
    )


def make_map_fn(split, args):
    """
    Convert AIME rows to framework format:
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


def create_file_name(args, split):
    fpart = "wsp" if args.system_prompt else "ns"
    return f"aime2025_processed_{args.run_id}_{fpart}_{split}.parquet"


def split_dataset_for_training(dataset, args):
    """
    Create train/val/test datasets from available splits.
    Priority:
    1) Existing train split (+ optional test split).
    2) Fall back to test split when train does not exist.
    """
    if "train" in dataset:
        full_train = dataset["train"]
        train_val_split = full_train.train_test_split(
            test_size=args.val_ratio, seed=args.seed
        )
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        test_dataset = dataset["test"] if "test" in dataset else val_dataset
        return train_dataset, val_dataset, test_dataset

    if "test" in dataset:
        full = dataset["test"]
        test_split = full.train_test_split(test_size=args.test_ratio, seed=args.seed)
        test_dataset = test_split["test"]
        remaining = test_split["train"]

        adjusted_val_ratio = args.val_ratio / (1.0 - args.test_ratio)
        val_split = remaining.train_test_split(
            test_size=adjusted_val_ratio, seed=args.seed
        )
        train_dataset = val_split["train"]
        val_dataset = val_split["test"]
        return train_dataset, val_dataset, test_dataset

    raise ValueError(f"Unsupported dataset splits: {list(dataset.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="opencompass/AIME2025")
    parser.add_argument("--data_config", default="AIME2025-I")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="aime2025_v1")
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant. Think step-by-step and output the final answer after '####'.",
    )
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=123345)
    args = parser.parse_args()

    dataset = datasets.load_dataset(args.data_source, args.data_config)
    train_dataset, val_dataset, test_dataset = split_dataset_for_training(dataset, args)

    train_dataset = train_dataset.map(
        function=make_map_fn("train", args), with_indices=True, num_proc=args.num_proc
    )
    val_dataset = val_dataset.map(
        function=make_map_fn("val", args), with_indices=True, num_proc=args.num_proc
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", args), with_indices=True, num_proc=args.num_proc
    )

    os.makedirs(args.local_dir, exist_ok=True)
    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_file_name = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))

    train_dataset.to_parquet(train_file_name)
    val_dataset.to_parquet(val_file_name)
    test_dataset.to_parquet(test_file_name)

    print("\n\n\n")
    print(train_dataset[0])
    print(val_dataset[0])
    print(test_dataset[0])
    print("\n\n\n")

    print(f"Train file: {train_file_name} with {len(train_dataset)} examples.")
    print(f"Val file: {val_file_name} with {len(val_dataset)} examples.")
    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
    print("Done.")
