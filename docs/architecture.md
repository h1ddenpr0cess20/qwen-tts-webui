# Architecture

This document explains how the system is wired together and where key responsibilities live.

## High-level flow

1. The client sends a `/api/tts` request with text and mode.
2. The backend resolves the model and device, loads the model (cached), and synthesizes audio.
3. The result is returned as a WAV stream.
4. Optional: the client requests MP3 conversion or video rendering using the WAV data URL.

## Backend modules

- `app/main.py`
  - Creates the FastAPI app, sets up CORS, mounts static frontend assets.
- `app/api/routes.py`
  - Defines API endpoints and response types.
- `app/schemas.py`
  - Pydantic request schemas and validation.
- `app/config.py`
  - Environment variables, defaults, and storage paths.
- `app/services/model_loader.py`
  - Model loader with LRU cache and dtype selection.
- `app/services/tts_synthesizer.py`
  - Mode-specific synthesis logic and chunking.
- `app/services/voice_profiles.py`
  - Profile creation, listing, storage, and deletion.
- `app/services/audio_conversion.py`
  - WAV data URL to MP3 conversion via `pydub`.
- `app/services/video_renderer.py`
  - FFmpeg-based waveform/spectrum/pulse video rendering.
- `app/utils/*`
  - Shared helpers for audio decoding, text wrapping, and file cleanup.

## Text chunking

`/api/tts` enforces a 500-character limit per chunk:

- When `chunk_text=true`, the text is split on word boundaries.
- When `chunk_text=false`, requests over 500 characters fail.

Chunked audio is concatenated and returned as a single WAV stream.

## Voice profiles

Voice profiles store a model-generated prompt so you can reuse a cloned voice without re-uploading audio.

- Stored in `voices/` by default (configurable via `VOICE_PROFILE_DIR`).
- Saved as `.pt` plus `.meta.json` for faster listings.
- The server enforces that a profile is used with the same model it was created with.

## CORS and security

The server enables permissive CORS (`allow_origins=["*"]`) and does not implement authentication. Consider tightening CORS and adding auth if you deploy beyond localhost.
