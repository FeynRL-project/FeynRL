import argparse
import io
import json
import os
import datasets
from PIL import Image

"""
MathVista preprocessing (AI4Math/MathVista -> parquet shards).

By default this mirrors the official Hugging Face dataset layout and writes
two shards:
  - mathvista_testmini.parquet
  - mathvista_test.parquet

Run with the environment that has `datasets` installed, e.g.:
  conda activate feynrl-upgrade
  python data_prep/mathvista.py --local_dir /path/to/mathvista
"""


SYSTEM_PROMPT = "You are a helpful assistant. Answer the question based on the image provided."


def pil_to_png_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def get_choice_letter(choices: list, answer: str) -> str:
    """Map choice text to A/B/C/D. Returns the raw answer if not found in choices."""
    try:
        return chr(ord("A") + choices.index(answer))
    except (ValueError, TypeError):
        return answer


def _normalized_answer(example: dict, idx: int) -> str:
    """Return a non-empty ground-truth label or raise with context."""
    for key in ("answer", "solution"):
        raw_value = example.get(key)
        value = "" if raw_value is None else str(raw_value).strip()
        if value != "":
            return value

    split = example.get("split", "unknown")
    pid = example.get("pid", "unknown")
    raise ValueError(
        f"MathVista sample idx={idx} pid={pid} split={split} has no usable label in "
        "'answer' or 'solution'."
    )


def _maybe_normalized_answer(example: dict) -> str:
    """Return a trimmed label when present, otherwise an empty string."""
    for key in ("answer", "solution"):
        raw_value = example.get(key)
        value = "" if raw_value is None else str(raw_value).strip()
        if value != "":
            return value
    return ""


def build_solution(example: dict) -> str:
    """Serialize ground-truth metadata as a JSON string for the reward function."""
    qt = example["question_type"]
    raw_answer = _normalized_answer(example, example.get("index", -1))
    at = example.get("answer_type") or "text"
    precision = example.get("precision")

    if qt == "multi_choice":
        letter = get_choice_letter(example.get("choices") or [], raw_answer)
        return json.dumps({"answer": letter, "question_type": "multi_choice"})
    else:
        return json.dumps({
            "answer": str(raw_answer).strip(),
            "question_type": "free_form",
            "answer_type": at,
            "precision": float(precision) if precision is not None else 1.0,
        })


def build_sft_answer(example: dict) -> str:
    """Short answer text that ImagePairedFeed appends during SFT training."""
    qt = example["question_type"]
    raw_answer = _normalized_answer(example, example.get("index", -1))
    if qt == "multi_choice":
        return get_choice_letter(example.get("choices") or [], raw_answer)
    return str(raw_answer).strip()


def build_solution_or_empty(example: dict) -> str:
    """Return serialized solution metadata when labeled, else an empty string."""
    if _maybe_normalized_answer(example) == "":
        return ""
    return build_solution(example)


def build_sft_answer_or_empty(example: dict) -> str:
    """Return short answer text when labeled, else an empty string."""
    if _maybe_normalized_answer(example) == "":
        return ""
    return build_sft_answer(example)


def create_prompt(example: dict, system_prompt: str) -> list:
    """Build chat messages using the pre-formatted query (includes hints + choices)."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example["query"]})
    return messages


def make_map_fn(split: str, params):
    def process_fn(example: dict, idx: int) -> dict:
        pil = example.get("decoded_image") or example.get("image") or example.get("images")
        # Some dataset versions store the image feature under different keys.
        # The HF datasets Image feature decodes to a PIL.Image.Image on access.
        if pil is None:
            raise ValueError(f"Missing image payload (expected 'decoded_image' or 'image') at index {idx}")
        example_with_index = dict(example)
        example_with_index["index"] = idx
        return {
            "prompt": create_prompt(example, params.system_prompt),
            "image_bytes": pil_to_png_bytes(pil),
            "answer": build_sft_answer_or_empty(example_with_index),
            "solution": build_solution_or_empty(example_with_index),
            "question_type": example["question_type"],
            "answer_type": example.get("answer_type") or "text",
            "split": split,
            "index": idx,
        }
    return process_fn


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="AI4Math/MathVista")
    parser.add_argument(
        "--local_dir",
        default="",
        help="Output directory for parquet shards.",
    )
    parser.add_argument("--system_prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    keep = {"prompt", "image_bytes", "answer", "solution",
            "question_type", "answer_type", "split", "index"}

    os.makedirs(args.local_dir, exist_ok=True)
    split_to_output = {
        "testmini": os.path.join(args.local_dir, "mathvista_testmini.parquet"),
        "test": os.path.join(args.local_dir, "mathvista_test.parquet"),
    }

    for split_name, output_path in split_to_output.items():
        ds = datasets.load_dataset(args.data_source, split=split_name)
        ds = ds.map(make_map_fn(split_name, args), with_indices=True, num_proc=args.num_proc)
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        ds.to_parquet(output_path)
        print(f"{split_name:<8}: {output_path}  ({len(ds)} examples)")

    print("Done.")
