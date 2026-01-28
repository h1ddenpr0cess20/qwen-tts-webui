"""Dependency imports with a friendly error if missing."""

try:
    import torch
    from qwen_tts import Qwen3TTSModel
except Exception as exc:  # pragma: no cover - import guard for missing deps
    raise RuntimeError("qwen-tts and torch are required. Install from requirements.txt.") from exc

__all__ = ["torch", "Qwen3TTSModel"]
