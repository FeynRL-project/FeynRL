import argparse
import os
import datasets

def parse_hh_conversation(text):
    '''
        Parse Anthropic HH conversation text into structured turns.

        Returns:
            List[Tuple[str, str]]  -> [(role, content), ...]
            role are {"user", "assistant"}
    '''
    turns = []
    current_role = None
    buffer = []

    for line in text.splitlines():
        line = line.rstrip()

        if line.startswith("Human:"):
            if buffer:
                turns.append((current_role, "\n".join(buffer).strip()))
                buffer = []
            current_role = "user"
            buffer.append(line[len("Human:"):].strip())

        elif line.startswith("Assistant:"):
            if buffer:
                turns.append((current_role, "\n".join(buffer).strip()))
                buffer = []
            current_role = "assistant"
            buffer.append(line[len("Assistant:"):].strip())

        else:
            buffer.append(line)

    if buffer:
        turns.append((current_role, "\n".join(buffer).strip()))

    return turns


# ============================================================
# Turn Alternation Enforcement
# ============================================================

def enforce_strict_alternation(turns):
    """
    Ensure turns strictly alternate between user and assistant.
    If consecutive turns have the same role, merge their content.

    Example:
        user, user, assistant  ->  user(merged), assistant
    """

    if not turns:
        return turns

    merged = []
    prev_role, prev_content = turns[0]

    flag = 0

    for role, content in turns[1:]:
        if role == prev_role:

            flag = 1
            # Merge consecutive same-role turns
            prev_content = prev_content.rstrip() + "\n\n" + content.lstrip()
        else:
            merged.append((prev_role, prev_content))
            prev_role, prev_content = role, content

    merged.append((prev_role, prev_content))

    return merged


# ============================================================
# First-Divergence Splitting
# ============================================================

def split_at_first_divergence(chosen_turns, rejected_turns):
    '''
        Split two turn sequences at the first differing turn.

        Returns:
            prompt_turns
            chosen_continuation_turns
            rejected_continuation_turns
    '''
    min_len = min(len(chosen_turns), len(rejected_turns))

    for i in range(min_len):
        if chosen_turns[i] != rejected_turns[i]:
            return (
                chosen_turns[:i],
                chosen_turns[i:],
                rejected_turns[i:],
            )

    # If identical up to min_len, one may continue further
    return (
        chosen_turns[:min_len],
        chosen_turns[min_len:],
        rejected_turns[min_len:],
    )

def build_prompt_messages(prompt_turns, system_prompt=None):
    '''
        Build structured chat messages for IT models.
    '''
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for role, content in prompt_turns:
        if role: #check it role is not None
            messages.append({"role": role, "content": content}) 

    return messages

def render_continuation_text(turns):
    '''
        Convert continuation turns into raw assistant continuation text.

        We keep full trajectory after divergence.
        Chat template during training will handle role formatting.

        Returns:
            str
    '''
    text_parts = []

    for role, content in turns:
        if role == "assistant":
            text_parts.append(content.strip())
        elif role == "user":
            # Keep user turns in continuation (trajectory-level DPO)
            text_parts.append(content.strip())

    return "\n".join(text_parts).strip()

def make_map_fn(split, args):
    '''
        Map dataset examples to DPO format.
    '''
    def process_fn(example, idx):
        chosen_raw = example["chosen"]
        rejected_raw = example["rejected"]

        chosen_turns = parse_hh_conversation(chosen_raw)
        rejected_turns = parse_hh_conversation(rejected_raw)

        if not chosen_turns or not rejected_turns:
            return None

        prompt_turns, chosen_cont, rejected_cont = split_at_first_divergence(
            chosen_turns,
            rejected_turns,
        )

        # Enforce strict alternation
        prompt_turns = enforce_strict_alternation(prompt_turns)
        chosen_cont = enforce_strict_alternation(chosen_cont) 
        rejected_cont = enforce_strict_alternation(rejected_cont) 

        # Convert continuation turns to text
        chosen_text = render_continuation_text(chosen_cont)
        rejected_text = render_continuation_text(rejected_cont)

        # Must have non-empty prompt and continuations
        if not prompt_turns or not chosen_text.strip() or not rejected_text.strip():
            return None

        data = {
            "prompt": build_prompt_messages(prompt_turns, args.system_prompt),
            "answer": chosen_text,
            "rejected_answer": rejected_text,
            "split": split,
            "index": idx,
        }

        return data

    return process_fn

def create_file_name(args, split):
    fpart = "wsp" if args.system_prompt else "ns"
    return f"hh_rlhf_dpo_{args.run_id}_{fpart}_{split}.parquet"


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="Anthropic/hh-rlhf")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="multiturn")
    parser.add_argument("--system_prompt", default=None)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ########
    # Load Dataset
    ########
    dataset = datasets.load_dataset(args.data_source)
    full_train = dataset["train"]
    test_split = full_train.train_test_split(test_size=args.test_ratio, seed=args.seed)

    ########
    # Train / Val / Test Split
    ########
    remaining = test_split["train"]
    test_dataset = test_split["test"]

    val_split = remaining.train_test_split(test_size=args.val_ratio, seed=args.seed)

    train_dataset = val_split["train"]
    val_dataset = val_split["test"]

    ########
    # Map to DPO Format
    ########
    train_dataset = train_dataset.map(make_map_fn("train", args), with_indices=True, num_proc=args.num_proc,
                                     remove_columns=train_dataset.column_names).filter(lambda x: x is not None)

    val_dataset = val_dataset.map(make_map_fn("val", args), with_indices=True, num_proc=args.num_proc,
                                  remove_columns=val_dataset.column_names).filter(lambda x: x is not None)

    test_dataset = test_dataset.map(make_map_fn("test", args), with_indices=True, num_proc=args.num_proc,
                                    remove_columns=test_dataset.column_names).filter(lambda x: x is not None)

    ########
    # save dataset
    ########
    os.makedirs(args.local_dir, exist_ok=True)

    train_file = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_file   = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_file  = os.path.join(args.local_dir, create_file_name(args, "test"))

    train_dataset.to_parquet(train_file)
    val_dataset.to_parquet(val_file)
    test_dataset.to_parquet(test_file)

    # --------------------------------------------------------
    # Sanity Check Print
    # --------------------------------------------------------
    for isample in range(5):

        print(100 * "=")
        print(100 * "=")

        sample = train_dataset[isample]
        print(sample)

        print("\n" + "=" * 80)
        print("Sample Example")
        print("=" * 80)
        for m in sample["prompt"]:
            print(f"{m['role']}: {m['content']}\n")

        print("=" * 80)
        print("Chosen Continuation:\n")
        print(sample["answer"])
        print("=" * 80)
        print("Rejected Continuation:\n")
        print(sample["rejected_answer"])
        print("=" * 80)

        print(f"\nTrain: {len(train_dataset)}")
        print(f"Val:   {len(val_dataset)}")
        print(f"Test:  {len(test_dataset)}")
        print("Done.")
