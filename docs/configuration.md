# Configuration

Configuration is driven by environment variables. Defaults are defined in `app/config.py`.

## Environment variables

- `QWEN_TTS_MODEL`
  - Default model ID or local path.
  - Default: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
- `QWEN_TTS_DEVICE`
  - Device map string (for example: `cuda:0`, `cpu`, `mps`).
  - Default: `cuda:0` if CUDA is available, otherwise `cpu`.
- `QWEN_TTS_USE_FLASH`
  - Set to `1`, `true`, or `yes` to enable FlashAttention 2.
- `QWEN_TTS_CUSTOM_MODEL`
  - Overrides the default model for Custom Voice mode.
  - Default: uses `QWEN_TTS_MODEL`.
- `QWEN_TTS_VD_MODEL`
  - Overrides the default model for Voice Design.
  - Default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`.
- `QWEN_TTS_CLONE_MODEL`
  - Overrides the default model for Voice Clone.
  - Default: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
- `QWEN_TTS_VIDEO_FONT`
  - Full path to a font file for video transcript rendering.
  - Useful for CJK/foreign scripts on Windows.
- `VOICE_PROFILE_DIR`
  - Optional override for where voice profiles are stored.
  - Default: `voices/` at the repo root.

## Model loading and caching

- Models are loaded via `Qwen3TTSModel.from_pretrained`.
- A small LRU cache keeps up to three model instances alive (`app/services/model_loader.py`).
- Dtype is chosen per device:
  - `bfloat16` for CUDA or MPS.
  - `float32` for CPU.

## Storage layout

Voice profiles are stored under `VOICE_PROFILE_DIR`:

```
voices/
  my_voice.pt
  my_voice.meta.json
```

The `.pt` file contains the cloning prompt plus metadata. The `.meta.json` file stores a subset for faster listing.

## Frontend asset location

The backend serves the static UI from `frontend/` when the directory exists. If you remove or rename the folder, the API will still work, but the UI will not be served.
