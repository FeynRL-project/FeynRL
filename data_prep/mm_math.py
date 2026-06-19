import argparse
import io
import json
import os
import re
import zipfile

import datasets
from huggingface_hub import hf_hub_download
from PIL import Image
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig, parse


def pil_to_png_bytes(pil_image):
    '''
       Encode a PIL image to raw PNG bytes (decoded downstream by image_utils.load_images).
    '''
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


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


def normalize_question(question):
    '''
       Remove the dataset's inline "<image>" marker and keep only the raw question. The data
       feed (ImagePairedFeed) injects the model-correct image placeholder itself, so the prompt
       text must NOT contain "<image>".
    '''
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return question.replace("<image>", "", 1).strip()


def get_image(example, idx):
    for key in ("image", "images", "decoded_image"):
        value = example.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            if len(value) == 0:
                continue
            return value[0]
        return value
    raise ValueError(f"MM_Math sample idx={idx} has no usable image field")


def load_source_examples(repo_id: str) -> list[dict]:
    '''
       Download the raw MM_Math jsonl + image zip directly from the dataset
       repo and return normalized in-memory examples.
    '''
    jsonl_path = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="MM_Math/MM_Math.jsonl")
    zip_path   = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename="MM_Math/MM_Math.zip")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        member_map = {name: name for name in members}
        basename_map = {}
        for name in members:
            basename_map.setdefault(os.path.basename(name), name)

        examples = []
        for idx, row in enumerate(rows):
            file_name = row.get("file_name")
            if not file_name:
                raise ValueError(f"MM_Math sample idx={idx} missing file_name")

            member = member_map.get(file_name) or basename_map.get(os.path.basename(file_name))
            if member is None:
                raise FileNotFoundError(f"Could not find image '{file_name}' in MM_Math.zip")

            with zf.open(member) as img_f:
                pil = Image.open(img_f).convert("RGB")
                image = pil.copy()

            normalized = dict(row)
            normalized["image"] = image
            examples.append(normalized)

    return examples

def normalize_solution(example: dict, idx: int) -> str:
    raw  = example.get("solution")
    text = "" if raw is None else str(raw).strip()
    if not text:
        raise ValueError(f"MM_Math sample idx={idx} has empty solution")
    return text


def _extract_last_boxed(text: str) -> str | None:
    matches = []
    start = 0
    while True:
        boxed_start = text.find("\\boxed{", start)
        if boxed_start == -1:
            break
        i = boxed_start + len("\\boxed{")
        depth = 1
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            matches.append(text[boxed_start + len("\\boxed{"): i - 1].strip())
            start = i
        else:
            break
    return matches[-1] if matches else None


def _extract_by_anchor(text: str) -> str | None:
    patterns = (
        r"(?is)(?:final answer|answer)\s*[:：]\s*(.+)$",
        r"(?is)(?:therefore|thus|so|hence)\s*,?\s*(.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            candidate = match.group(1).strip()
            if candidate:
                return candidate
    return None


def extract_final_answer(solution_text: str) -> str:
    '''
       Extract the final answer for reward verification while keeping the full
       worked solution separately for SFT.
    '''
    boxed = _extract_last_boxed(solution_text)
    if boxed:
        return boxed

    anchored = _extract_by_anchor(solution_text)
    if anchored:
        solution_text = anchored

    for target in ((LatexExtractionConfig(),), (ExprExtractionConfig(),)):
        try:
            parsed = parse(solution_text, target)
        except Exception:
            parsed = []
        if parsed:
            return str(parsed[-1]).strip()

    lines = [line.strip() for line in solution_text.splitlines() if line.strip()]
    if lines:
        return lines[-1]
    raise ValueError("could not extract final answer from solution")


def make_map_fn(split, params):
    '''
       This function reads data and returns a dictionary, mirroring gsm8k.py / geometry3k.py
       columns plus the VLM image:
         prompt   : chat message(s) with the plain question (image placeholder added by the feed)
         answer   : full worked solution (training target for SFT, data.answer_key)
         solution : final answer only (evaluation target for math_verify, data.solution_key)
         image    : raw PNG bytes (decoded by image_utils.load_images; data.image_key)
    '''
    def process_fn(example, idx):
        question     = normalize_question(example["question"])
        full_solution = normalize_solution(example, idx)
        final_answer  = extract_final_answer(full_solution)
        pil           = get_image(example, idx)
        return {"prompt": create_prompt(question, params.system_prompt),
                "answer": full_solution,    # training target (SFT)
                "solution": final_answer,   # evaluation target (math_verify)
                "image": pil_to_png_bytes(pil),
                "year": example.get("year"),
                "difficult": example.get("difficult"),
                "knowledge": example.get("knowledge"),
                "split": split,
                "index": idx,
               }

    return process_fn

def create_file_name(params, split):
    '''
       This function creates file name based on the params.
    '''
    fpart = 'wsp' if params.system_prompt else 'ns'
    file_name = f"mm_math_processed_{params.run_id}_{fpart}_{split}.parquet"
    return file_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="THU-KEG/MM_Math")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--system_prompt", default="You are a helpful assistant. Solve the math problem shown in the image.")
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source_split", default="train")
    args = parser.parse_args()

    if args.val_size <= 0 or args.test_size <= 0:
        raise ValueError("val_size and test_size must both be > 0")

    source_examples = load_source_examples(args.data_source)
    source_ds = datasets.Dataset.from_list(source_examples)
    required = args.val_size + args.test_size + 1
    if len(source_examples) < required:
        raise ValueError(f"Source split is too small for fixed splits: need at least {required} rows, got {len(source_examples)}")

    train_val_ds, test_ds = source_ds.train_test_split(test_size=args.test_size, seed=args.seed).values()
    train_ds, val_ds = train_val_ds.train_test_split(
        test_size=args.val_size,
        seed=args.seed,
    ).values()

    # Columns consumed downstream (prompt/answer/solution/image) plus mm_math metadata.
    keep = {"prompt",
            "answer",
            "solution",
            "image",
            "year",
            "difficult",
            "knowledge",
            "split",
            "index",
            }

    train_ds = train_ds.map(make_map_fn("train", args), with_indices=True, num_proc=args.num_proc)
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep])

    val_ds = val_ds.map(make_map_fn("val", args), with_indices=True, num_proc=args.num_proc)
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep])

    test_ds = test_ds.map(make_map_fn("test", args), with_indices=True, num_proc=args.num_proc)
    test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in keep])

    os.makedirs(args.local_dir, exist_ok=True)
    train_path = os.path.join(args.local_dir, create_file_name(args, "train"))
    val_path   = os.path.join(args.local_dir, create_file_name(args, "val"))
    test_path  = os.path.join(args.local_dir, create_file_name(args, "test"))

    train_ds.to_parquet(train_path)
    val_ds.to_parquet(val_path)
    test_ds.to_parquet(test_path)

    print(f"Train file: {train_path} with {len(train_ds)} examples.")
    print(f"Val file: {val_path} with {len(val_ds)} examples.")
    print(f"Test file: {test_path} with {len(test_ds)} examples.")
    print("Done.")