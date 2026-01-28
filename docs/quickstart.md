# Quickstart

This guide covers local setup, required dependencies, and how to run the app.

## Prerequisites

- Python 3.10 or newer.
- A GPU-capable PyTorch build is strongly recommended.
- Disk and bandwidth for model downloads (several GB on first run).
- Optional but recommended:
  - FlashAttention 2 for supported GPUs.
  - `ffmpeg` for MP3 conversion and video rendering.

## Install Python dependencies

```bash
pip install -r requirements.txt
```

Optional FlashAttention (only for compatible GPUs):

```bash
pip install -U flash-attn --no-build-isolation
```

## Model downloads (optional)

By default, models are pulled on demand by `qwen-tts`. If your machine cannot download models at runtime, pre-download a model and point `QWEN_TTS_MODEL` to a local path.

Example:

```bash
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice
```

Then set:

```bash
setx QWEN_TTS_MODEL "C:\path\to\Qwen3-TTS-12Hz-1.7B-CustomVoice"
```

## Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

- UI: `http://localhost:8000`
- API root: `/api/*`

## Basic workflow

1. Open the UI and select a mode (Custom Voice, Voice Design, or Voice Clone).
2. Enter text and optional settings (language, speaker, persona, etc.).
3. Click Generate to synthesize audio.
4. Download WAV or convert to MP3.
5. (Optional) Export a waveform/spectrum video.

## Environment overview

Common environment variables are documented in `docs/configuration.md`.
