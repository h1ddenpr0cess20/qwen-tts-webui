"""Centralized configuration values for the TTS service."""

import os
from pathlib import Path

from app.deps import torch

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
VOICE_PROFILE_DIR = Path(os.getenv("VOICE_PROFILE_DIR", BASE_DIR / "voices"))
VOICE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_ID = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_DEVICE = os.getenv("QWEN_TTS_DEVICE")
if not DEFAULT_DEVICE:
    DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DEFAULT_CUSTOM_MODEL = os.getenv("QWEN_TTS_CUSTOM_MODEL", DEFAULT_MODEL_ID)
DEFAULT_VOICE_DESIGN_MODEL = os.getenv(
    "QWEN_TTS_VD_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
DEFAULT_CLONE_MODEL = os.getenv("QWEN_TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

USE_FLASH = os.getenv("QWEN_TTS_USE_FLASH", "").lower() in {"1", "true", "yes"}

VIDEO_LAYOUTS = {
    "vertical": (1080, 1920),
    "square": (1080, 1080),
    "landscape": (1920, 1080),
}
