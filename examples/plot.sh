#!/usr/bin/env bash
# Sample usage for examples/generate_plots.py
# Run from the examples/ directory: bash plot.sh

# List all available experiment keys:
python generate_plots.py --list

# Plot all experiments with defaults:
python generate_plots.py

# Plot a single experiment:
python generate_plots.py --experiment math_qwen3_4b
python generate_plots.py --experiment math_qwen2_5_1b
python generate_plots.py --experiment healthbench_qwen2_5_1b

# Plot multiple experiments:
python generate_plots.py --experiment math_qwen3_4b --experiment math_qwen2_5_1b

# Override the smoothing window (1 = raw, no smoothing):
python generate_plots.py --smooth 10
python generate_plots.py --smooth 1

# Truncate x-axis (minutes for time-series, steps for step-series):
python generate_plots.py --experiment math_qwen3_4b --max-x 120
python generate_plots.py --experiment math_qwen2_5_1b --max-x 30

# Write all PNGs to a custom directory:
python generate_plots.py --output-dir /tmp/feynrl_plots

# Combine flags:
python generate_plots.py --experiment math_qwen3_4b --smooth 3 --max-x 220
