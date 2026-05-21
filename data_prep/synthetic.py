"""
Synthetic micro-dataset generator in FeynRL Parquet format.

Produces a Parquet file with schema:
  prompt : list[{role, content}]  (system optional, user required)
  answer : str

Use cases:
  - offline smoke tests (no network access required)
  - local dev without downloading a full dataset
  - real datasets: swap _BASE_EXAMPLES for an actual data reader
"""
import argparse
import os
import pandas as pd


_BASE_EXAMPLES = [
    {"question": "What is 1 + 1?",                              "answer": "2"},
    {"question": "Name the capital of France.",                  "answer": "Paris"},
    {"question": "What color is the sky on a clear day?",        "answer": "Blue"},
    {"question": "What is 3 * 4?",                              "answer": "12"},
    {"question": "What is the boiling point of water in Celsius?", "answer": "100"},
    {"question": "What planet is closest to the Sun?",           "answer": "Mercury"},
    {"question": "What is 10 - 3?",                             "answer": "7"},
    {"question": "Name one primary color.",                      "answer": "Red"},
    {"question": "How many sides does a triangle have?",         "answer": "3"},
    {"question": "What is the square root of 16?",              "answer": "4"},
    {"question": "Which gas do plants absorb from the air?",    "answer": "Carbon dioxide"},
    {"question": "What is 100 divided by 4?",                   "answer": "25"},
    {"question": "What continent is Brazil on?",                "answer": "South America"},
    {"question": "What is the chemical symbol for water?",      "answer": "H2O"},
    {"question": "How many minutes are in one hour?",           "answer": "60"},
    {"question": "What is 2 to the power of 3?",               "answer": "8"},
]


def build_synthetic_dataframe(n: int = 8, system_prompt: str | None = None) -> pd.DataFrame:
    base = _BASE_EXAMPLES
    examples = (base * ((n // len(base)) + 1))[:n]
    rows = []
    for ex in examples:
        if system_prompt:
            prompt = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": ex["question"]},
            ]
        else:
            prompt = [{"role": "user", "content": ex["question"]}]
        rows.append({"prompt": prompt, "answer": ex["answer"]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic micro-dataset in FeynRL Parquet format."
    )
    parser.add_argument("--output",        required=True,       help="Output .parquet path")
    parser.add_argument("--n",             type=int, default=8, help="Number of examples")
    parser.add_argument("--system_prompt", default=None,        help="Optional system prompt")
    args = parser.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df = build_synthetic_dataframe(n=args.n, system_prompt=args.system_prompt)
    df.to_parquet(args.output, index=False)
    print(f"Wrote {len(df)} examples to {args.output}.")
