# Installation Guide

## Prerequisites

Before beginning, ensure you have:

- An NVIDIA GPU (Ampere architecture or newer recommended for full feature support)
- NVIDIA drivers compatible with CUDA 12.x (driver version ≥ 525.85 recommended)
- Conda (Miniconda or Anaconda) installed and initialized in your shell

**Verify your GPU and driver are visible:**

```bash
nvidia-smi
```

You should see a table showing your GPU model, driver version, and a CUDA version in the top-right corner. If this fails, fix your NVIDIA driver before proceeding — nothing else will work without it.

---

## Step 1: Create and Activate the Conda Environment

```bash
conda create -n leanrl-env python=3.13.1 -y
conda activate leanrl-env
```

---

## Step 2: Install the CUDA Toolkit

```bash
conda install -c nvidia cuda-toolkit==12.2 -y
```

**Verify the toolkit installed correctly:**

```bash
nvcc --version
```

You should see output referencing CUDA 12.2.

---

## Step 3: Install PyTorch with CUDA 12.6 Support

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**Verify PyTorch can see your GPU:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

All three lines should return a version string, `True`, and your GPU name respectively. **Do not proceed if `cuda.is_available()` returns `False`.**

---

## Step 4: Install Core Packages

```bash
pip install \
  pydantic==2.12.5 \
  pyyaml==6.0.3 \
  huggingface_hub==0.36.0 \
  transformers==4.57.6 \
  deepspeed==0.18.6 \
  datasets==4.5.0 \
  vllm==0.15.1 \
  ray==2.53.0 \
  mlflow==3.8.1 \
  wandb==0.25.0
```

**Verify critical imports:**

```bash
python -c "import vllm; import deepspeed; import transformers; print('All core imports OK')"
```

---

## Step 5: Install FlashAttention (Last — Always)

```bash
pip install flash-attn --no-build-isolation --config-settings="--jobs=8" --verbose
```

**Verify FlashAttention:**

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```