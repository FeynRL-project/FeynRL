# FeynRL Dockerfile
#
# Usage:
#   docker build -t feynrl:latest .
#
# Run (single node, 8 GPUs):
#   docker run --gpus all --ipc=host --net=host \
#     -v /hot-data:/hot-data \
#     -v /ceph:/ceph \
#     feynrl:latest \
#     python main_rl.py --config configs/rl_args.yaml
#
# Slurm (enroot):
#   enroot import dockerd://feynrl:latest
#   sbatch your_job.sh

FROM nvidia/cuda:12.6.3-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
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

# Install uv for fast dependency management
RUN pip install -U pip uv

WORKDIR /FeynRL

# Create venv
ENV VIRTUAL_ENV=/FeynRL/.venv
RUN uv venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --index-strategy unsafe-best-match -r requirements.txt

# Install flash-attn (improves vLLM and training performance)
RUN uv pip install flash-attn --no-build-isolation

# Copy project source
COPY . /FeynRL

ENTRYPOINT []
CMD ["/bin/bash"]
