# Security Policy

This project is intended to run on a trusted machine or a private network. It has **no authentication** and enables permissive CORS by default, so do not expose it directly to the internet without additional safeguards.

## Reporting vulnerabilities
- Preferred: open a private GitHub security advisory for this repo.
- If advisory submission isn’t available, open a minimal issue titled `[SECURITY]` without exploit details; the maintainer will follow up for a private channel.
- Please include reproduction steps, logs, and environment details (OS, GPU, CUDA/cuDNN versions, Python version).

## Deployment hardening
- **Network boundary**: run behind a reverse proxy or VPN. Add authentication at the proxy if you must expose it beyond localhost.
- **HTTPS**: terminate TLS at your proxy; do not serve plaintext over the public internet.
- **CORS**: `app/main.py` currently sets `allow_origins=["*"]`. Restrict this to your own origins (or remove `CORSMiddleware`) before exposing the API.
- **Uploads**: `/api/voice_profiles/import` accepts `.pt` files. Only import profiles you trust; treat them like code.
- **Disk writes**: voice profiles are stored under `voices/` (configurable via `VOICE_PROFILE_DIR`). Keep this directory on trusted storage and clear it if sensitive material was uploaded.
- **Models and binaries**: model downloads and FFmpeg use native code; keep them updated and fetch from trusted sources (e.g., official Hugging Face repos).

## Dependency and runtime hygiene
- Keep Python dependencies current (`pip install -r requirements.txt --upgrade`), especially FastAPI/Starlette, Torch, and audio/video tooling.
- Prefer running under least privilege (non-root containers/users).
- If you build Docker images, avoid publishing images that contain cached voice profiles or API keys.

## Scope
- This policy covers the FastAPI backend, the bundled vanilla JS frontend, and local scripts in this repository.
- No guarantee of long-term support is provided; security fixes are handled on a best-effort basis.
