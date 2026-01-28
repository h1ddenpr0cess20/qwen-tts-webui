# Qwen3 TTS Web App

A FastAPI + vanilla JS UI to run [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) locally: custom voices, voice design, voice cloning, and per-request model selection.

## Prerequisites

- Python 3.10+ with a GPU-enabled PyTorch build (GPU strongly recommended).
- Disk/bandwidth for model downloads (several GB on first load).
- Optional: FlashAttention 2 if your GPU supports it (`pip install -U flash-attn --no-build-isolation`).

## Setup

```bash
pip install -r requirements.txt
```

If your machine cannot download weights during runtime, pre-download a model (e.g. `huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local-dir ./Qwen3-TTS-12Hz-1.7B-CustomVoice`) and point `QWEN_TTS_MODEL` to that path.

## Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000` for the UI. API endpoints live under `/api/*`.

## Configuration (env vars)

- `QWEN_TTS_MODEL` — default model id or local path (default: `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`).
- `QWEN_TTS_DEVICE` — device map (default: `cuda:0` if available, else `cpu`).
- `QWEN_TTS_USE_FLASH` — set to `1` to try FlashAttention 2.
- `QWEN_TTS_CUSTOM_MODEL` — override default for Custom Voice mode (else uses `QWEN_TTS_MODEL`).
- `QWEN_TTS_VD_MODEL` — override default for Voice Design mode (default: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`).
- `QWEN_TTS_CLONE_MODEL` — override default for Voice Clone mode (default: `Qwen/Qwen3-TTS-12Hz-1.7B-Base`).
- `QWEN_TTS_VIDEO_FONT` — full path to a font file for video transcript rendering (useful for CJK/foreign text).

Requests can override `model_id` and `device` per call, but the UI auto-selects the recommended models per mode from the upstream README.

### Model quick reference (from upstream README)
- Custom Voice: `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice` (speaker list included).
- Voice Design: `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` (describe persona; no speaker list).
- Voice Clone: `Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base` (provide ref audio + transcript).
- Tokenizer (encode/decode only): `Qwen/Qwen3-TTS-Tokenizer-12Hz`.

## Features

- **Custom Voice**: pick a provided speaker, language, and optional style prompt.
- **Voice Design**: describe a persona and language; the model invents the voice.
- **Voice Clone**: supply a reference audio (URL/path/base64) plus transcript to clone a voice.
- **Model selection**: choose any released model id or local directory per request.
- **UI**: shows available speakers/languages, plays inline, and offers WAV download.
- **Recording/Upload for cloning**: record in-browser or upload; the UI converts to WAV before sending.
- **Saved voices**: build a reusable voice profile (clone prompt) once and reuse it without re-uploading audio.
- **MP3 download**: generation stays WAV; pick MP3 in the UI to convert the generated clip on demand (requires `pydub` + `ffmpeg` available).
- **Video export**: render a vertical/square/landscape MP4 with waveform/spectrum visuals and transcript (requires `ffmpeg` with `drawtext`).

## API Examples

### Custom Voice
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -o custom.wav \
  -d '{
    "mode": "custom_voice",
    "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "language": "English",
    "speaker": "Ryan",
    "instruct": "Energetic podcast intro with a smile.",
    "text": "Welcome back to our weekend build session. Grab your coffee and let us ship!"
  }'
```

### Voice Design
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -o design.wav \
  -d '{
    "mode": "voice_design",
    "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "language": "English",
    "instruct": "Late-night radio host, warm baritone, unhurried pace with soft consonants.",
    "text": "You are tuned to 88.5 FM. Outside the city is sleeping, but we are still here with you."
  }'
```

### Voice Clone
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -o clone.wav \
  -d '{
    "mode": "voice_clone",
    "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "language": "English",
    "ref_audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
    "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.",
    "text": "This is a cloned voice reading a new paragraph. We can keep the tone calm and measured."
  }'
```

For quick experiments without a transcript, set `"x_vector_only_mode": true` and omit `ref_text` (quality may drop).

### Save a voice profile (reuse clone prompt)
```bash
curl -X POST http://localhost:8000/api/voice_profiles \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_radio_host",
    "model_id": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "ref_audio": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-TTS-Repo/clone.wav",
    "ref_text": "Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you."
  }'
```

Then synthesize with that cached prompt:
```bash
curl -X POST http://localhost:8000/api/tts \
  -H "Content-Type: application/json" \
  -o clone_with_profile.wav \
  -d '{
    "mode": "voice_clone",
    "voice_profile": "my_radio_host",
    "text": "We can keep reusing this voice without re-uploading audio.",
    "language": "English"
  }'
```

### Voice Design → Clone Reuse

1) Use the Voice Design model to synthesize a short clip with the desired persona.  
2) Feed that clip and its text back as `ref_audio`/`ref_text` with `mode: "voice_clone"` using the Base model.  
This keeps a consistent designed voice for longer scripts.

## Frontend

The UI exposes the same options: pick mode, enter model id/path, language, speaker (custom voice), style (voice design), or ref audio/transcript (voice clone). It streams back a WAV, plays inline, and offers a download link.

## Notes

- GPU + bfloat16/float16 greatly reduces latency and memory; CPU runs will be slow.
- Reference audio can be a public URL, local path, or base64 data URI. Keep it clean and ~3–10s for best cloning.
- The page pulls a Google Font; remove the `<link>` in `frontend/index.html` if you need offline-only assets.
