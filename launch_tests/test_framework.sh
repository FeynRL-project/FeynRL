#!/bin/bash

# Configuration
TEST_CONFIG_DIR="./launch_tests/configs"
LOG_DIR="./logs/launch_tests/results"
mkdir -p "$LOG_DIR"

# Algorithms to test
ALGS=("sgrpo" "cispo" "ppo" "dpo" "sft")

# Refresh configs from abstract bases
python launch_tests/utils/generate_test_configs.py
sudo python launch_tests/utils/create_dummy_data.py

# Collect results for summary
declare -a SUMMARY_TABLE

run_test() {
    local alg=$1
    echo "=========================================================="
    echo "Testing Algorithm: $alg"
    echo "=========================================================="
    
    local config_file="$TEST_CONFIG_DIR/${alg}.yaml"
    if [ ! -f "$config_file" ]; then
        echo "Error: Config file $config_file not found!"
        SUMMARY_TABLE+=("$alg | N/A | N/A | N/A | MISSING_CONFIG")
        return 1
    fi

    local log_file="$LOG_DIR/${alg}_test.log"
    echo "Logging to: $log_file"

    # Extract metadata for summary
    local stage=$(python -c "import yaml; print(yaml.safe_load(open('$config_file'))['deepspeed']['zero_optimization']['stage'])" 2>/dev/null || echo "??")
    local logger=$(python -c "import yaml; print(yaml.safe_load(open('$config_file'))['run']['logger_type'])" 2>/dev/null || echo "??")
    local offload=$(python -c "import yaml; c=yaml.safe_load(open('$config_file')); print(c.get('model', {}).get('ref_model_offload_to_cpu', 'N/A'))" 2>/dev/null || echo "N/A")
    local batch_size=$(python -c "import yaml; c=yaml.safe_load(open('$config_file')); print(c.get('train', {}).get('train_batch_size_per_gpu', '??'))" 2>/dev/null || echo "??")
    local opt=$(python -c "import yaml; c=yaml.safe_load(open('$config_file')); print(c.get('train', {}).get('optimizer_name', '??'))" 2>/dev/null || echo "??")
    local lr=$(python -c "import yaml; c=yaml.safe_load(open('$config_file')); print(c.get('train', {}).get('lr', '??'))" 2>/dev/null || echo "??")
    local epochs=$(python -c "import yaml; c=yaml.safe_load(open('$config_file')); print(c.get('train', {}).get('total_number_of_epochs', '??'))" 2>/dev/null || echo "??")

    local status="SUCCESS"
    local start_time=$(date +%s)
    if [ "$alg" == "dpo" ]; then
        sudo -E "$TORCHRUN" --nproc_per_node=1 main_cl.py \
            --config-file "$config_file" \
            --experiment_id "test_dpo_$(date +%Y%m%d_%H%M%S)" \
            2>&1 | tee "$log_file"
    elif [ "$alg" == "sft" ]; then
        echo "Launching SFT test using torchrun and main_sl.py..."
        sudo -E "$TORCHRUN" --nproc_per_node=1 main_sl.py \
            --config-file "$config_file" \
            --experiment_id "test_sft_$(date +%Y%m%d_%H%M%S)" \
            2>&1 | tee "$log_file"
    else
        sudo -E "$PYTHON" main_rl.py \
            --config-file "$config_file" \
            --experiment_id "test_${alg}_$(date +%Y%m%d_%H%M%S)" \
            2>&1 | tee "$log_file"
    fi

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo "SUCCESS: $alg test completed."
    else
        echo "FAILURE: $alg test failed. Check $log_file for details."
        status="FAILURE"
    fi
    
    # Store result in table: Algorithm | Stage | Logger | Offload | Batch | Opt | LR | Epochs | Duration | Status
    SUMMARY_TABLE+=("$alg|$stage|$logger|$offload|$batch_size|$opt|$lr|$epochs|${duration}s|$status")
}

if [ -z "$1" ]; then
    echo "Usage: $0 [alg_name | all]"
    echo "Supported algorithms: ${ALGS[*]}"
    exit 1
fi

if [ "$1" == "all" ]; then
    for alg in "${ALGS[@]}"; do
        run_test "$alg"
    done
else
    run_test "$1"
fi

# Print Summary Table
echo ""
echo "======================================================================================================================================"
echo "                                                      EXPERIMENTAL TEST SUMMARY"
echo "======================================================================================================================================"
printf "%-10s | %-5s | %-8s | %-12s | %-6s | %-10s | %-8s | %-6s | %-8s | %-10s\n" "Algorithm" "Stage" "Logger" "Ref Offload" "Batch" "Optimizer" "LR" "Epochs" "Time" "Status"
echo "--------------------------------------------------------------------------------------------------------------------------------------"
for line in "${SUMMARY_TABLE[@]}"; do
    IFS='|' read -r salg sstage slogger soffload sbatch sopt slr sepochs sduration sstatus <<< "$line"
    printf "%-10s | %-5s | %-8s | %-12s | %-6s | %-10s | %-8s | %-6s | %-8s | %-10s\n" "$salg" "$sstage" "$slogger" "$soffload" "$sbatch" "$sopt" "$slr" "$sepochs" "$sduration" "$sstatus"
done
echo "======================================================================================================================================"
echo ""

# Cleanup artifacts to restore workspace status
echo "Cleaning up generated artifacts (configs, data, checkpoints)..."
sudo rm -rf ./launch_tests/results/
sudo rm -rf ./launch_tests/data
# Remove only the generated top-level configs, preserving the base/ directory
sudo rm -f ./launch_tests/configs/*.yaml
# Also cleanup any leftover wandb/mlrun metadata if present in root
sudo rm -rf ./wandb/
sudo rm -rf ./mlruns/
echo "Workspace restored. Logs are preserved in $LOG_DIR"
