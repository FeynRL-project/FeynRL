# FeynRL Dockerfile (default: FlashAttention v2, Ampere/Ada/pre-Hopper)
#
# Build (from repo root):
#   docker build -f docker/default/Dockerfile -t feynrl:default .
#
# Run (single node, 8 GPUs):
#   docker run --gpus all --ipc=host --net=host \
#     -v $HOME/.cache/huggingface:/root/.cache/huggingface \
#     -w /FeynRL \
#     feynrl:default \
#     python main_rl.py --config_file configs/rl_args.yaml
#
# See docs/HOWTO.md for SFT/RL examples and Slurm (enroot) usage.

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    net-tools \
    libibverbs-dev \
    librdmacm-dev \
    ibverbs-utils \
    rdmacm-utils \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    tmux \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip uv

WORKDIR /FeynRL

ENV VIRTUAL_ENV=/FeynRL/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

COPY requirements.txt .
RUN uv pip install --index-strategy unsafe-best-match -r requirements.txt

# FlashAttention v2 — pinned to the version exercised in CI/smoke tests.
# For Hopper/Blackwell (H100/B200), build docker/hopper/Dockerfile instead.
RUN uv pip install flash-attn==2.8.3 --no-build-isolation

COPY . /FeynRL

ENTRYPOINT []
CMD ["/bin/bash"]
