# RunPod Serverless Worker for Stable Audio VAE
# Uses official RunPod PyTorch base image (pre-cached on RunPod for fast starts)

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set environment
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install RunPod SDK
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

# Download VAE model from HuggingFace (official Stable Audio Open)
RUN curl -L -o /models/model.ckpt \
    "https://huggingface.co/stabilityai/stable-audio-open-1.0/resolve/main/vae_model.ckpt" && \
    curl -L -o /models/model_config.json \
    "https://huggingface.co/stabilityai/stable-audio-open-1.0/resolve/main/vae_model_config.json"

# Set model paths
ENV VAE_CONFIG_PATH=/models/model_config.json
ENV VAE_CKPT_PATH=/models/model.ckpt

# Pre-compile UMAP/numba
RUN python -c "import umap; print('UMAP ready')"

# Entry point
CMD ["python", "-u", "/app/rp_handler.py"]
