import argparse
import io
import os

import datasets
from PIL import Image


def pil_to_png_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def normalize_question(problem: str) -> str:
    '''
       Remove the dataset's inline image marker and keep only the raw question.
    '''
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError("problem must be a non-empty string")
    return problem.replace("<image>", "", 1).strip()


def create_prompt(question: str, system_prompt: str) -> list[dict[str, str]]:
    '''
       Build chat messages from the raw question text.
       By default, Geometry3K uses no system prompt.
    '''
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    return messages


def get_first_image(example: dict, idx: int):
    images = example.get("images")
    if not isinstance(images, list) or len(images) == 0:
        raise ValueError(f"Geometry3K sample idx={idx} has no images")
    return images[0]


def make_map_fn(split: str, system_prompt: str):
    def process_fn(example: dict, idx: int) -> dict:
        question = normalize_question(example["problem"])
        image = get_first_image(example, idx)
        answer = str(example["answer"]).strip()
        if not answer:
            raise ValueError(f"Geometry3K sample idx={idx} has empty answer")

        return {
            "prompt": create_prompt(question, system_prompt),
            "question": question,
            "image_bytes": pil_to_png_bytes(image),
            "solution": answer,
            "split": split,
            "index": idx,
        }

    return process_fn


def create_file_name(split: str) -> str:
    return f"geometry3k_{split}.parquet"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="hiyouga/geometry3k")
    parser.add_argument("--local_dir", required=True)
    parser.add_argument(
        "--system_prompt",
        default="",
        help="Optional system prompt. Default keeps only the raw question.",
    )
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    dataset = datasets.load_dataset(args.data_source)
    split_map = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    keep = {"prompt", "question", "image_bytes", "solution", "split", "index"}

    os.makedirs(args.local_dir, exist_ok=True)

    for source_split, output_split in split_map.items():
        split_ds = dataset[source_split]
        split_ds = split_ds.map(
            function=make_map_fn(output_split, args.system_prompt),
            with_indices=True,
            num_proc=args.num_proc,
        )
        split_ds = split_ds.remove_columns([c for c in split_ds.column_names if c not in keep])

        output_path = os.path.join(args.local_dir, create_file_name(output_split))
        split_ds.to_parquet(output_path)
        print(f"{source_split} -> {output_path} ({len(split_ds)} examples)")

    print("Done.")
