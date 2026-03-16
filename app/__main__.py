"""Allow running the app with ``python -m app`` or the ``qwen-tts`` console script."""

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3 TTS Web App")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
