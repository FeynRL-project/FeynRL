import argparse
import os
import re
import datasets
import pandas as pd

# Keys match the "Shared Evaluation Protocol" table in examples/llm/README.md.
# Each benchmark has a fixed HuggingFace source, so it's hard-coded in the
# LOADERS table below rather than taken as a CLI argument.
BENCHMARKS = [
    "gsm8k",
    "aime_2024",
    "aime_2025",
    "aime_2026",
    "amc",
    "amo_bench",
    "brumo_2025",
    "hmmt_feb_25",
    "hmmt_nov_25",
    "olympiad",
]

DEFAULT_SYSTEM_PROMPT = ("You are a helpful math assistant. Solve the problem step by step, "
                          "then give your final answer as a number")


def create_prompt(question, system_prompt):
    '''
       This creates general message with or without system prompt.
    '''
    if system_prompt:
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    return [{"role": "user", "content": question}]


def clean_answer(value):
    '''
       Strips a raw ground-truth value down to the bare expression
       math_verify_reward_func should grade: unwraps a whole-string \\boxed{...}
       and surrounding $ delimiters, but otherwise leaves the LaTeX untouched.
    '''
    text = str(value).strip()
    boxed = re.fullmatch(r"\\boxed\{(.*)\}", text, re.DOTALL)
    if boxed:
        text = boxed.group(1).strip()
    return text.strip("$").strip()


def extract_gsm8k_solution(answer_raw):
    '''
       GSM8K answers end in "#### <value>"; this pulls out just the value.
    '''
    match = re.search(r"####\s*(-?[0-9.,]+)", answer_raw)
    assert match is not None
    return match.group(1).replace(",", "").replace("$", "").replace("\n", "")


def finalize(problems, solutions):
    '''
       Builds the common problem/answer/solution/split/index frame shared by
       every benchmark. "answer" isn't graded (math_verify_reward_func only
       reads "solution") so it's just a "#### <solution>" marker kept for
       consistency with the SFT-oriented data_prep scripts.
    '''
    n = len(problems)
    return pd.DataFrame({
        "problem": list(problems),
        "answer": [f"#### {s}" for s in solutions],
        "solution": list(solutions),
        "split": ["test"] * n,
        "index": list(range(n)),
    })


def load_gsm8k():
    df = datasets.load_dataset("openai/gsm8k", "main")["test"].to_pandas()
    solutions = [extract_gsm8k_solution(a) for a in df["answer"]]
    return finalize(df["question"], solutions)


def load_aime_2024():
    df = datasets.load_dataset("HuggingFaceH4/aime_2024")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_aime_2025():
    df = datasets.load_dataset("MathArena/aime_2025")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_aime_2026():
    df = datasets.load_dataset("MathArena/aime_2026")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_amc():
    # AI-MO/aimo-validation-amc no longer carries 2024 problems; rawsh/2024_AMC12
    # is the 2024 AMC 12A+12B set the "amc" benchmark actually targets.
    df = datasets.load_dataset("rawsh/2024_AMC12")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_amo_bench():
    df = datasets.load_dataset("meituan-longcat/AMO-Bench")["test"].to_pandas()
    return finalize(df["prompt"], [clean_answer(a) for a in df["answer"]])


def load_brumo_2025():
    df = datasets.load_dataset("MathArena/brumo_2025")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_hmmt_feb_25():
    df = datasets.load_dataset("MathArena/hmmt_feb_2025")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_hmmt_nov_25():
    df = datasets.load_dataset("MathArena/hmmt_nov_2025")["train"].to_pandas()
    return finalize(df["problem"], [clean_answer(a) for a in df["answer"]])


def load_olympiad():
    # OE_TO_maths_en_COMP (text-only) + OE_MM_maths_en_COMP (multimodal) are the
    # English math competition subsets; image fields are dropped since this is a
    # text-only pipeline.
    configs = ["OE_TO_maths_en_COMP", "OE_MM_maths_en_COMP"]
    frames = [datasets.load_dataset("Hothan/OlympiadBench", c)["train"].to_pandas() for c in configs]
    df = pd.concat(frames, ignore_index=True)
    solutions = [clean_answer(", ".join(str(a) for a in answers)) for answers in df["final_answer"]]
    return finalize(df["question"], solutions)


LOADERS = {
    "gsm8k": load_gsm8k,
    "aime_2024": load_aime_2024,
    "aime_2025": load_aime_2025,
    "aime_2026": load_aime_2026,
    "amc": load_amc,
    "amo_bench": load_amo_bench,
    "brumo_2025": load_brumo_2025,
    "hmmt_feb_25": load_hmmt_feb_25,
    "hmmt_nov_25": load_hmmt_nov_25,
    "olympiad": load_olympiad,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downloads and packs the 10 Shared Evaluation "
                                                  "Protocol benchmarks (examples/llm/README.md) "
                                                  "directly from HuggingFace into --local_dir.")
    parser.add_argument("--local_dir", default="./data",
                        help="Output directory for the packed {benchmark}_test.parquet files.")
    parser.add_argument("--variant", choices=["ns", "wsp"], default="ns",
                        help="ns = no system prompt, wsp = with system prompt.")
    parser.add_argument("--benchmarks", nargs="+", default=BENCHMARKS, choices=BENCHMARKS,
                        help="Subset of benchmarks to pack (default: all 10).")
    args = parser.parse_args()

    system_prompt = DEFAULT_SYSTEM_PROMPT if args.variant == "wsp" else ""
    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Downloading and packing {len(args.benchmarks)} benchmark(s) into {args.local_dir} "
          f"(variant={args.variant})\n")

    results = []
    for benchmark in args.benchmarks:
        df = LOADERS[benchmark]()
        df["prompt"] = [create_prompt(q, system_prompt) for q in df["problem"]]
        out_df = df[["prompt", "answer", "solution", "split", "index"]]

        if out_df["prompt"].isnull().any() or out_df["solution"].isnull().any():
            raise ValueError(f"{benchmark}: null values in 'prompt' or 'solution'")

        out_path = os.path.join(args.local_dir, f"{benchmark}_test.parquet")
        out_df.to_parquet(out_path)

        # round-trip check to catch any parquet-write/schema bug
        check_df = pd.read_parquet(out_path)
        if len(check_df) != len(out_df):
            raise AssertionError(f"Row count mismatch writing {benchmark}: {len(out_df)} vs {len(check_df)}")
        if check_df["solution"].astype(str).tolist() != out_df["solution"].astype(str).tolist():
            raise AssertionError(f"Solution mismatch writing {benchmark}: packing altered ground-truth answers")

        print(f"[OK] {benchmark:12s} {len(out_df):5d} rows  -> {out_path}")
        results.append((benchmark, len(out_df)))

    print("\nAll benchmarks downloaded, packed, and verified successfully.")
    print(f"Total test examples: {sum(n for _, n in results)}")
