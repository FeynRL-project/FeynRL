import argparse
import os
import re
import datasets

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
       In gsm8k, answers end with #### <answer>.
       If we construct our answers this way or if the dataset has them, 
       this function will work.
    '''
    solution = re.search(r"####\s*(-?[0-9.,]+)", solution_str)
    if solution is None:
        return None
    final_solution = solution.group(1).replace(",", "").replace("$", "").replace("\n", "")
    return final_solution

def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary in the framework's format.
       allenai/Dolci-RL-Zero-Math-7B has:
       - 'prompt': The question string.
       - 'ground_truth': The final numeric answer string.
       - 'messages': List of message dicts (usually just the user prompt).
    '''
    def process_fn(example, idx):
        # The framework expects 'prompt', 'answer', 'solution', 'split', 'index'.
        question = example.get("prompt")
        # In case 'prompt' is missing but 'messages' is there
        if not question and "messages" in example and len(example["messages"]) > 0:
            question = example["messages"][0]["content"]
        
        solution = example.get("ground_truth")
        
        # Construct 'answer' for SFT/RL consistency.
        # Since Dolci-RL-Zero-Math doesn't provide reasoning steps, 
        # we provide the final answer in the expected format.
        answer_raw = f"#### {solution}"
        
        data = {
            "prompt": create_prompt(question, params.system_prompt),
            "answer": answer_raw, # used for SFT
            "solution": solution, # used for RL/Eval
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
    file_name = f"dolci_processed_{params.run_id}_{fpart}_{split}.parquet"
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="allenai/Dolci-RL-Zero-Math-7B")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="dolci_v1")
    parser.add_argument("--system_prompt", default="You are a helpful assistant. Provide brief reasoning and output the final answer in the format '#### <answer>'.")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Ratio of training data to use for validation")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Ratio of training data to use for test")
    parser.add_argument("--seed", type=int, default=123345)
    args = parser.parse_args()

    ########
    # load dataset from huggingface
    ########
    # Dolci-RL-Zero-Math-7B has only a 'train' split
    dataset = datasets.load_dataset(args.data_source)
    full_dataset = dataset["train"]
    
    # Split into train, val, and test
    test_split = full_dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
    test_dataset = test_split["test"]
    remaining_dataset = test_split["train"]
    
    val_split = remaining_dataset.train_test_split(test_size=args.val_ratio, seed=args.seed)
    train_dataset = val_split["train"]
    val_dataset = val_split["test"]

    ########
    # map dataset
    ########
    train_dataset = train_dataset.map(function=make_map_fn("train", params=args), with_indices=True, num_proc=args.num_proc)
    val_dataset = val_dataset.map(function=make_map_fn("val", params=args), with_indices=True, num_proc=args.num_proc)
    test_dataset = test_dataset.map(function=make_map_fn("test", params=args), with_indices=True, num_proc=args.num_proc)

    ########
    # save dataset
    ########
    os.makedirs(args.local_dir, exist_ok=True)
    train_file_name = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_file_name   = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_file_name  = os.path.join(args.local_dir, create_file_name(args, "test"))
    
    train_dataset.to_parquet(train_file_name)
    val_dataset.to_parquet(val_file_name)
    test_dataset.to_parquet(test_file_name)

    # print samples:
    print("Messages: ", train_dataset[0]["prompt"])
    print(80 * "=")
    print("Answer: ", train_dataset[0]["answer"])
    print(80 * "=")
    print("Solution: ", train_dataset[0]["solution"])


    print(f"Train file: {train_file_name} with {len(train_dataset)} examples.")
    print(f"Val file: {val_file_name} with {len(val_dataset)} examples.")
    print(f"Test file: {test_file_name} with {len(test_dataset)} examples.")
    print("Done.")
