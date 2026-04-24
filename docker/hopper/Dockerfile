# FeynRL Dockerfile (Hopper/Blackwell: FlashAttention 4, for H100/B200)
#
# Build (from repo root):
#   docker build -f docker/hopper/Dockerfile -t feynrl:hopper .
#
# NOTE: FlashAttention 4 has not yet been validated against FeynRL's SFT/RL
# smoke tests — see PR #56 discussion. Treat this image as experimental until
# that validation lands.

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

# FlashAttention 4 — Hopper/Blackwell only.
RUN uv pip install flash-attn-4 --no-build-isolation

COPY . /FeynRL

ENTRYPOINT []
CMD ["/bin/bash"]
