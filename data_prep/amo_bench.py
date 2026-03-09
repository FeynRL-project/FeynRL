import argparse
import os

import datasets


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


def make_map_fn(split, args):
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


def create_file_name(args, split):
    fpart = "wsp" if args.system_prompt else "ns"
    return f"amo_bench_processed_{args.run_id}_{fpart}_{split}.parquet"


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="meituan-longcat/AMO-Bench")
    parser.add_argument("--data_config", default=None)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="amo_bench_v1")
    parser.add_argument(
        "--system_prompt",
        default="You are a helpful assistant. Think step-by-step and output the final answer after '####'.",
    )
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    if args.data_config:
        dataset = datasets.load_dataset(args.data_source, args.data_config)
    else:
        dataset = datasets.load_dataset(args.data_source)

    test_dataset = merge_splits_as_test(dataset)
    test_dataset = test_dataset.map(
        function=make_map_fn("test", args), with_indices=True, num_proc=args.num_proc
    )

    os.makedirs(args.local_dir, exist_ok=True)
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))

    test_dataset.to_parquet(test_file_name)

    print("\n\n\n")
    print(test_dataset[0])
    print("\n\n\n")

    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
    print("Done.")
