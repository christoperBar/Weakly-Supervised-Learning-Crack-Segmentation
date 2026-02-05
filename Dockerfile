FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set python 3.10 default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Core Python deps
RUN pip install \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    matplotlib>=3.7.0 \
    tqdm>=4.65.0 \
    Pillow>=10.0.0 \
    opencv-python>=4.8.0

# PyTorch (CUDA 11.8)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Optional DenseCRF
RUN pip install git+https://github.com/lucasb-eyer/pydensecrf.git

CMD ["/bin/bash"]
