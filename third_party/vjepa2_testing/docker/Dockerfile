# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG UID=1000
ARG GID=1000

# System setup
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-dev \
        python3-venv \
        build-essential \
        git \
        curl \
        ca-certificates \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
        rsync \
        unzip \
        pkg-config && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip && \
    python -m pip install --upgrade pip setuptools wheel && \
    rm -rf /var/lib/apt/lists/*

ENV TORCH_CUDA_ARCH_LIST="8.9+PTX"
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV DEBIAN_FRONTEND=dialog

WORKDIR /workspace

COPY docker/requirements.txt /tmp/container-requirements.txt
RUN pip install --no-cache-dir -r /tmp/container-requirements.txt && rm /tmp/container-requirements.txt

COPY vjepa2 /workspace/vjepa2

CMD ["/bin/bash"]
