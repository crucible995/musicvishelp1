# RunPod Serverless Worker for Stable Audio VAE
# Optimized for L40 GPU with fast cold starts

FROM nvidia/cuda:12.1-cudnn8-runtime-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install --upgrade pip && \
    pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install RunPod SDK and dependencies
RUN pip install runpod>=1.6.0

# Install stable-audio-tools from GitHub
RUN pip install git+https://github.com/Stability-AI/stable-audio-tools.git

# Install additional dependencies
RUN pip install \
    umap-learn>=0.5.0 \
    numpy>=1.24.0 \
    scipy>=1.10.0 \
    numba>=0.57.0

# Create models directory
RUN mkdir -p /models

# Copy handler
COPY rp_handler.py /app/rp_handler.py

# Copy model files (baked into image for fast cold starts)
COPY stable_audio_2_0_vae.json /models/
COPY sao_vae_tune_100k_unwrapped.ckpt /models/

# Set model paths
ENV VAE_CONFIG_PATH=/models/stable_audio_2_0_vae.json
ENV VAE_CKPT_PATH=/models/sao_vae_tune_100k_unwrapped.ckpt

# Pre-download UMAP dependencies (numba compilation)
RUN python -c "import umap; print('UMAP ready')"

# Entry point
CMD ["python", "-u", "/app/rp_handler.py"]
