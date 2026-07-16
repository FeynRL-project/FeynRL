import argparse
import os
import re
import datasets
import pandas as pd

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
    # Shared eval uses only the held-out HF test shard so it stays disjoint from
    # data_prep/gsm8k.py, which now derives train/val only from the HF train shard.
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
    # Shared eval uses the 2024 AMC 12A+12B benchmark from this dataset card.
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


def get_loaders():
    # Keys match the "Shared Evaluation Protocol" table in examples/llm/README.md.
    # Each benchmark has a fixed HuggingFace source, so the mapping is kept here
    # rather than exposed as a free-form CLI argument.
    return {
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


def get_benchmarks():
    return list(get_loaders().keys())


def create_file_name(params, benchmark):
    '''
       This function creates file name based on the params.
    '''
    fpart = 'wsp' if params.system_prompt else 'ns'
    return f"{benchmark}_processed_{params.run_id}_{fpart}_test.parquet"


if __name__ == "__main__":
    benchmarks = get_benchmarks()
    loaders = get_loaders()
    parser = argparse.ArgumentParser(description="Downloads and packs the 10 Shared Evaluation "
                                                  "Protocol benchmarks (examples/llm/README.md) "
                                                  "directly from HuggingFace into --local_dir.")
    parser.add_argument("--local_dir", default="./data",
                        help="Output directory for the packed {benchmark}_processed_{run_id}_{ns|wsp}_test.parquet files.")
    parser.add_argument("--run_id", default="123245")
    parser.add_argument("--system_prompt", default="",
                        help="Optional system prompt prepended to every packed prompt.")
    parser.add_argument("--benchmarks", nargs="+", default=benchmarks, choices=benchmarks,
                        help="Subset of benchmarks to pack (default: all 10).")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    print(f"Downloading and packing {len(args.benchmarks)} benchmark(s) into {args.local_dir} "
          f"(system_prompt={'set' if args.system_prompt else 'empty'})\n")

    results = []
    for benchmark in args.benchmarks:
        df = loaders[benchmark]()
        df["prompt"] = [create_prompt(q, args.system_prompt) for q in df["problem"]]
        out_df = df[["prompt", "answer", "solution", "split", "index"]]

        if out_df["prompt"].isnull().any() or out_df["solution"].isnull().any():
            raise ValueError(f"{benchmark}: null values in 'prompt' or 'solution'")

        out_path = os.path.join(args.local_dir, create_file_name(args, benchmark))
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
