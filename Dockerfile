FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        build-essential \
        git \
        ninja-build \
        ffmpeg \
        libsndfile1 \
        sox \
        libsox-fmt-all \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python3 -m pip install --upgrade pip

ARG TORCH_CUDA=cu121
RUN python3 -m pip install --index-url https://download.pytorch.org/whl/${TORCH_CUDA} torch torchvision torchaudio

RUN python3 -m pip install -r requirements.txt

ARG INSTALL_FLASH_ATTN=1
RUN if [ "$INSTALL_FLASH_ATTN" = "1" ]; then python3 -m pip install -U flash-attn --no-build-isolation; fi

COPY app ./app
COPY frontend ./frontend
COPY docs ./docs
COPY README.md ./

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
