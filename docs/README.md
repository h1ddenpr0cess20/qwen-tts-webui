# Qwen3 TTS Documentation

This folder contains detailed documentation for the Qwen3 TTS web app. Start with the Quickstart if you want to run the app locally, then dive into the API, frontend, and architecture docs for implementation details.

## Contents

- `docs/quickstart.md` - Install, configure, and run the app.
- `docs/api.md` - REST API endpoints, payloads, and error behavior.
- `docs/configuration.md` - Environment variables, model selection, and storage paths.
- `docs/frontend.md` - UI behavior and how it interacts with the API.
- `docs/architecture.md` - Module overview, runtime flow, and caching.
- `docs/troubleshooting.md` - Common issues and fixes.

## At a glance

- Backend: FastAPI serving the API and static frontend.
- Models: `qwen-tts` models loaded on demand with a small LRU cache.
- Outputs: WAV streaming for TTS, MP3 conversion (optional), and MP4 video rendering (optional).
- Storage: Voice profiles stored under `voices/` by default.
