import argparse
import os
import re
import datasets
# adopted based on https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py

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

def extract_solution(solution_str):
    '''
       This extracts solution from the answer.
    '''
    solution = re.search(r"####\s*(-?[0-9.,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(1).replace(",", "").replace("$", "").replace("\n", "")
    return final_solution

def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary.
       An example of this data is:
       {'question': 'James writes a 3-page letter to 2 different friends twice a week.  How many pages does he write a year?',
        'answer': 'He writes each friend 3*2=<<3*2=6>>6 pages a week\nSo he writes 6*2=<<6*2=12>>12 pages every week\nThat means
        he writes 12*52=<<12*52=624>>624 pages a year\n#### 624'}
    '''
    def process_fn(example, idx):
        question   = example.pop("question")
        answer_raw = example.pop("answer")
        solution   = extract_solution(answer_raw)
        data       = {"prompt": create_prompt(question, params.system_prompt),
                      "answer": answer_raw, # this will be used for training which contains the both training traces and final answer after ####.
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
    file_name = f"gsm8k_processed_{params.run_id}_{fpart}_{split}.parquet"
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="openai/gsm8k")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--system_prompt", default="You are a helpful assistant. Think step-by-step and output the final answer after '####'.")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of training data to use for validation")
    parser.add_argument("--seed", type=int, default=123345)
    args = parser.parse_args()

    ########
    # load dataset from huggingface
    ########
    dataset = datasets.load_dataset(args.data_source, "main")
    # split train into train and val
    train_val_split = dataset["train"].train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = train_val_split["train"]
    val_dataset   = train_val_split["test"]
    # load test dataset
    test_dataset  = dataset["test"]

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

    ########
    # print samples from each shard
    ########
    SEP = "=" * 72
    THIN = "-" * 72
    N_SAMPLES = 2

    for split_name, ds in [("TRAIN", train_dataset), ("VAL", val_dataset), ("TEST", test_dataset)]:
        print(f"\n{SEP}")
        print(f"  {split_name} SAMPLES  ({N_SAMPLES} of {len(ds)})")
        print(SEP)
        for i in range(N_SAMPLES):
            ex = ds[i]
            # Extract messages from the prompt
            sys_msg  = next((m["content"] for m in ex["prompt"] if m["role"] == "system"), None)
            user_msg = next((m["content"] for m in ex["prompt"] if m["role"] == "user"), "")
            print(f"\n[{split_name} #{ex['index']}]")
            print(THIN)
            if sys_msg:
                print(f"SYSTEM:\n  {sys_msg}\n")
            print(f"QUESTION:\n  {user_msg}")
            print(f"\nSOLUTION: {ex['solution']}")
            print(THIN)
        print()

    print("Done.")
    