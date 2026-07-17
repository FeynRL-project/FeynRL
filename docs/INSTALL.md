# Installation Guide

## Prerequisites

Before beginning, ensure you have:

- An NVIDIA GPU (Ampere architecture or newer recommended for full feature support)
- NVIDIA drivers compatible with your fleet ceiling
- Conda (Miniconda or Anaconda) installed and initialized in your shell

**Verify your GPU and driver are visible:**

```bash
nvidia-smi
```

You should see a table showing your GPU model, driver version, and a CUDA version in the top-right corner. If this fails, fix your NVIDIA driver before proceeding — nothing else will work without it.

---

FeynRL has been tested on nodes capped at:

- CUDA 12.2 on A100s
- CUDA 12.4 on H100s

The install sequence below matches that environment build.

---

## Step 1: Create and Activate the Conda Environment

We recommend **Python 3.12** for better compatibility with packages used in FeynRL. Any Python version in the `>=3.10, <=3.12` range should also work.

```bash
conda create -n feynrl-env python=3.12 -y
conda activate feynrl-env
python -m pip install --upgrade pip setuptools wheel
```

---

## Step 2: Install PyTorch

Start from the conservative CUDA 12.1 PyTorch wheels:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
```

---

## Step 3: Install the Base Python Dependencies

Install the pinned base packages from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

---

## Step 4: Install the CUDA Toolkit

Install a CUDA toolkit inside the Conda env so DeepSpeed can find `nvcc` at `ENV/bin/nvcc`:

```bash
conda install -c nvidia cuda-toolkit=12.2 -y
```

Verify the toolkit installed correctly:

```bash
nvcc --version
```

---

## Step 5: Install `vllm`

Install `vllm` last:

```bash
python -m pip install vllm==0.19.1
```

`vllm==0.19.1` rewrites the torch runtime underneath the env. The expected final runtime is `torch==2.10.0+cu128`.

---

## Step 6: Verify the Environment

Verify the final stack:

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -c "import vllm; print(vllm.__version__)"
python -c "import transformers; print(transformers.__version__)"
python -c "import deepspeed; print(deepspeed.__version__)"
python -c "import peft; print(peft.__version__)"
python -c "import datasets; print(datasets.__version__)"
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
python -c "import ray; print(ray.__version__)"
python -c "import mlflow; print(mlflow.__version__)"
python -c "import wandb; print(wandb.__version__)"
python -c "import pydantic; print(pydantic.__version__)"
python -c "import yaml; print(yaml.__version__)"
python -c "import importlib.metadata as m; print(m.version('math-verify'))"
which nvcc
```

Verify critical imports:

```bash
python -c "from misc.model_loading import build_hf_model; print('misc.model_loading import OK')"
python -c "import algs.RL.common; print('algs.RL.common import OK')"
```

---

## Step 7: Authenticate with Hugging Face and Weights & Biases

Many models (e.g. Llama, Gemma) require accepting a license on the Hugging Face Hub before you can download them. Log in so you can access gated models for training and evaluation:

```bash
huggingface-cli login
```

You will be prompted to paste a token from [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

If you use **Weights & Biases** as your experiment tracker (`logger_type: "wandb"` in your config), log in so training metrics are reported correctly:

```bash
wandb login
```

You will be prompted to paste an API key from [https://wandb.ai/authorize](https://wandb.ai/authorize).

---

For runtime issues, node-specific loader issues, or scaling problems, see the **[Troubleshooting Guide](../docs/TROUBLESHOOTING.md)**.
