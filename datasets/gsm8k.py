import argparse
import os
import re
import datasets
# adopted based on https://github.com/volcengine/verl/blob/main/examples/data_preprocess/gsm8k.py

def create_prompt(question, use_system_prompt):
    '''
       This creates general message with or without system prompt.
    '''
    if use_system_prompt:
        system_prompt = 'You are a helpful assistant. The goal is to solve the math problem. To solve the problem, think step by step and output the final answer after "####".'
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
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution

def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary.
    '''
    def process_fn(example, idx):
        question   = example.pop("question")
        answer_raw = example.pop("answer")
        solution   = extract_solution(answer_raw)
        data       = {
            "prompt": create_prompt(question, params.use_system_prompt),\
            "answer": solution,
            "reward_model": {"style": "rule", "ground_truth": solution},
            "split": split,
            "index": idx,
        }
        return data

    return process_fn

def create_file_name(params, split):
    '''
       This function creates file name based on the params.
    '''
    fpart = 'wsp' if params.use_system_prompt == True else 'ns' 
    file_name = f"gsm8k_{params.run_id}_{fpart}_{split}.parquet"
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="openai/gsm8k")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--use_system_prompt", default=False)
    parser.add_argument("--num_proc", default=4)
    args = parser.parse_args()
    
    ########
    # load dataset from huggingface
    ########
    dataset = datasets.load_dataset(args.data_source, "main")
    train_dataset = dataset["train"]
    test_dataset  = dataset["test"]

    ########
    # map dataset
    ########
    train_dataset = train_dataset.map(function=make_map_fn("train", params=args), with_indices=True, num_proc=args.num_proc)
    test_dataset = test_dataset.map(function=make_map_fn("test", params=args), with_indices=True, num_proc=args.num_proc)

    ########
    # save dataset
    ########
    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    test_file_name = os.path.join(args.local_dir, create_file_name(args, "test"))
    train_dataset.to_parquet(train_file_name)
    test_dataset.to_parquet(test_file_name)
    
    print("\n")
    print(f"Train file: {train_file_name} with {len(train_dataset)} examples.")
    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
    print("Done.")
    