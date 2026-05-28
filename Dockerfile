# 基于 NVIDIA CUDA 12.8 镜像构建
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 安装必要的系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    yasm \
    nasm \
    ffmpeg \
    libgl1 \
    python3 \
    python3-pip \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 设置 Python 环境
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 安装 PyTorch 和相关依赖
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装额外的机器学习依赖
RUN pip install transformers accelerate datasets evaluate lm_eval sentencepiece opencv-python imageio
