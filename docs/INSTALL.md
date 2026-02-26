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

We recommend **Python 3.13.1** for better compatibility with packages used in FeynRL. However, other versions of Python 3.13 should also work.

```bash
conda create -n feynrl-env python=3.13.1 -y
conda activate feynrl-env
```

---

## Step 2: Install the CUDA Toolkit

While we recommend **CUDA 12.2** (the version we tested), any **CUDA toolkit ≥ 12.2** should work as long as it’s compatible with your GPU and your installed PyTorch build. FeynRL does not rely on CUDA-toolkit version–specific features, so newer compatible versions are generally fine.

```bash
conda install -c nvidia cuda-toolkit==12.2 -y
```

**Verify the toolkit installed correctly:**

```bash
nvcc --version
```

---

## Step 3: Install PyTorch and Core Packages

For cleaner dependency management, core packages and PyTorch are listed in `requirements.txt`. To ensure we install PyTorch with GPU support that is compatible with CUDA 12.x, the `requirements.txt` file specifies the correct index URL.

```bash
pip install -r requirements.txt
```

**Verify PyTorch can see your GPU:**

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

All three lines should return a version string, `True`, and your GPU name respectively. **Do not proceed if `cuda.is_available()` returns `False`.**

**Verify critical imports:**

```bash
python -c "import vllm; import deepspeed; import transformers; print('All core imports OK')"
```

---

## Step 4: Install FlashAttention

Flash Attention is often the most fragile dependency to install and it is slower to install than other packages. It should be built from source using the exact command below to ensure it compiles against your environment's CUDA toolkit and PyTorch versions correctly.

```bash
pip install flash-attn==2.8.3 --no-build-isolation --config-settings="--jobs=8" --verbose
```

**Verify FlashAttention:**

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
```