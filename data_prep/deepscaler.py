import argparse
import os
import datasets

# adopted based on data_prep/gsm8k.py

def create_prompt(question, system_prompt):
    '''
       This creates general message with or without system prompt.
    '''
    if system_prompt:
        message = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question}
                  ]

    else:
        message = [
                    {"role": "user", "content": question}
                  ]

    return message

def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary.
       DeepScaleR's raw examples separate the worked derivation ("solution")
       from the bare final answer ("answer"), unlike GSM8K which encodes both
       in a single "#### <value>" suffixed string. This reconstructs that same
       GSM8K-style "answer" field (full derivation + "#### <value>" marker,
       used for SFT training targets) while "solution" stays the bare cleaned
       value used for reward grading.
       An example of the raw data is:
       {'problem': 'What is the smallest positive integer with exactly 12 positive integer divisors?',
        'answer': '96',
        'solution': ''}
    '''
    def process_fn(example, idx):
        question     = example.pop("problem")
        raw_answer   = example.pop("answer")
        raw_solution = example.pop("solution").strip()
        answer_raw   = f"{raw_solution}\n#### {raw_answer}" if raw_solution else f"#### {raw_answer}"
        solution     = raw_answer.replace(",", "").replace("$", "").replace("\n", "").strip()
        data         = {"prompt": create_prompt(question, params.system_prompt),
                        "answer": answer_raw, # this will be used for training which contains the reasoning trace (if any) and final answer after ####.
                        "solution": solution, # this will be used for evaluation.
                        "split": split,
                        "index": idx,
                        }
        return data

    return process_fn

def create_file_name(params, split):
    '''
       This function creates file name based on the params.
    '''
    fpart = 'wsp' if params.system_prompt else 'ns'
    file_name = f"deepscaler_preview_processed_{params.run_id}_{fpart}_{split}.parquet"
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="agentica-org/DeepScaleR-Preview-Dataset")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--system_prompt", default="You are a helpful math assistant. Solve the problem step by step, then give your final answer as a number")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of data to use for test")
    parser.add_argument("--seed", type=int, default=123345)
    args = parser.parse_args()

    ########
    # load dataset from huggingface (single "train" split — carve out val/test ourselves)
    ########
    dataset = datasets.load_dataset(args.data_source)

    train_valtest_split = dataset["train"].train_test_split(test_size=args.val_ratio + args.test_ratio, seed=args.seed)
    train_dataset = train_valtest_split["train"]
    val_test_split = train_valtest_split["test"].train_test_split(
        test_size=args.test_ratio / (args.val_ratio + args.test_ratio), seed=args.seed)
    val_dataset  = val_test_split["train"]
    test_dataset = val_test_split["test"]

    ########
    # map dataset
    ########
    train_dataset = train_dataset.map(function=make_map_fn("train", params=args), with_indices=True, num_proc=args.num_proc)
    val_dataset = val_dataset.map(function=make_map_fn("val", params=args), with_indices=True, num_proc=args.num_proc)
    test_dataset = test_dataset.map(function=make_map_fn("test", params=args), with_indices=True, num_proc=args.num_proc)

    ########
    # save dataset
    ########
    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_file_name   = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_file_name  = os.path.join(args.local_dir, create_file_name(args, "test"))
    train_dataset.to_parquet(train_file_name)
    val_dataset.to_parquet(val_file_name)
    test_dataset.to_parquet(test_file_name)

    print("\n")
    print(f"Train file: {train_file_name} with {len(train_dataset)} examples.")
    print(f"Val file: {val_file_name} with {len(val_dataset)} examples.")
    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
