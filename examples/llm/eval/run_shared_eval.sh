#!/usr/bin/env bash
# Runs the Shared Evaluation Protocol (see README.md) across all 10 benchmarks
# in sequence, reusing one base config (eval/eval_shared_base.yaml) for model/rollout
# params and swapping in each benchmark's test file and checkpoint dir.
#
# Prereq: python data_prep/shared_eval_benchmarks.py --local_dir ./data
#         (produces ./data/<benchmark>_test.parquet -- see README's Data Preparation section)
#
# Usage (run from the repo root):
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ./examples/llm/eval/run_shared_eval.sh \
#       --model ./ckps/qwen2.5-1.5b-instruct/checkpoint \
#       --experiment_id my_run
#
# Results land under ./ckps/eval/<experiment_id>/<benchmark>/rollout_stats.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
BASE_CONFIG="$SCRIPT_DIR/eval_shared_base.yaml"

BENCHMARKS=(gsm8k aime_2024 aime_2025 aime_2026 amc amo_bench brumo_2025 hmmt_feb_25 hmmt_nov_25 olympiad)

MODEL=""
EXPERIMENT_ID="shared_eval"
ROLLOUT_GPUS=1
DATA_DIR="./data"
CHECKPOINT_ROOT="./ckps/eval"

usage() {
    echo "Usage: $0 --model <path_or_hf_id> [--experiment_id NAME] [--rollout_gpus N] [--data_dir DIR] [--checkpoint_root DIR]"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --experiment_id) EXPERIMENT_ID="$2"; shift 2 ;;
        --rollout_gpus) ROLLOUT_GPUS="$2"; shift 2 ;;
        --data_dir) DATA_DIR="$2"; shift 2 ;;
        --checkpoint_root) CHECKPOINT_ROOT="$2"; shift 2 ;;
        -h|--help) usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

[[ -z "$MODEL" ]] && usage

cd "$REPO_ROOT"

TMP_CONFIG_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_CONFIG_DIR"' EXIT

for benchmark in "${BENCHMARKS[@]}"; do
    test_file="${DATA_DIR}/${benchmark}_test.parquet"
    if [[ ! -f "$test_file" ]]; then
        echo "[SKIP] $benchmark: missing $test_file (run data_prep/shared_eval_benchmarks.py first)" >&2
        continue
    fi

    config_path="${TMP_CONFIG_DIR}/${benchmark}.yaml"
    sed \
        -e "s#__MODEL_NAME__#${MODEL}#g" \
        -e "s#__CHECKPOINT_DIR__#${CHECKPOINT_ROOT}/${EXPERIMENT_ID}/${benchmark}#g" \
        -e "s#__ROLLOUT_GPUS__#${ROLLOUT_GPUS}#g" \
        -e "s#__TEST_FILE__#${test_file}#g" \
        -e "s#__EXPERIMENT_ID__#${EXPERIMENT_ID}_${benchmark}#g" \
        "$BASE_CONFIG" > "$config_path"

    exp_id="${EXPERIMENT_ID}_${benchmark}"
    echo "=== [$benchmark] experiment_id=${exp_id} ==="
    python main_eval.py --config_file "$config_path" --experiment_id "$exp_id"
done

echo "Shared evaluation protocol complete for experiment_id=${EXPERIMENT_ID}"
