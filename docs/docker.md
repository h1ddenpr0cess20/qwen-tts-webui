# Docker

This document describes how to build and run the app with Docker, plus common GPU/CPU considerations.

## Quick start

Build the image:

```bash
docker build -t qwen-tts .
```

Run the container:

```bash
docker run --rm -p 8000:8000 qwen-tts
```

Open the UI at `http://localhost:8000`.

## Docker Compose

Start the service:

```bash
docker compose up --build
```

The compose file defaults to GPU (`QWEN_TTS_DEVICE=cuda:0`). For CPU-only, change it to `cpu` in `docker-compose.yml`.

## GPU notes

- The Docker image is CUDA-enabled by default and uses a CUDA "devel" base image so `flash-attn` can compile.
- The container will only use a GPU if the host has an NVIDIA-enabled Docker setup and the container has access to GPU devices.
- If the UI shows CPU-only behavior, confirm `QWEN_TTS_DEVICE=cuda:0` and that `torch.cuda.is_available()` is true in the container.
- If you need a different CUDA build, rebuild with `--build-arg TORCH_CUDA=cu118` (or another supported tag) and a matching CUDA base image.
- The Docker image installs `flash-attn` by default. If you want to skip it, build with `--build-arg INSTALL_FLASH_ATTN=0`.

## Persisting voice profiles

Voice profiles are stored under `voices/` by default. To keep them across container runs, mount a volume:

```bash
docker run --rm -p 8000:8000 -v ${PWD}/voices:/app/voices qwen-tts
```

You can also set a different directory with `VOICE_PROFILE_DIR`.

## Using a local model directory

If you pre-download a model, mount it and point `QWEN_TTS_MODEL` at the local path:

```bash
docker run --rm -p 8000:8000 -v ${PWD}/models:/models -e QWEN_TTS_MODEL=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice qwen-tts
```

## Troubleshooting

- No audio or MP3/video export fails: verify `ffmpeg` is available (included in the image).
- Slow inference: use a GPU or a smaller model ID.
- CUDA not detected: ensure the host GPU drivers and NVIDIA container runtime are installed.
