"""FFmpeg-related helpers for video rendering."""

import os
from pathlib import Path
from typing import Optional

from app.utils.text import contains_cjk


def escape_ffmpeg_path(path: Path) -> str:
    """Escape a filesystem path for FFmpeg drawtext usage.

    Args:
        path: Path to escape.

    Returns:
        Escaped string suitable for FFmpeg filter strings.
    """

    return str(path).replace("\\", "/").replace(":", "\\:")


def find_system_font(text: str) -> Optional[Path]:
    """Find a suitable system font for the given text on Windows.

    Args:
        text: Transcript text that may contain CJK characters.

    Returns:
        Path to the first matching font, or None if not found or non-Windows.
    """

    env_font = os.getenv("QWEN_TTS_VIDEO_FONT", "").strip()
    if env_font:
        font_path = Path(env_font)
        if font_path.exists():
            return font_path
    if os.name != "nt":
        return None
    fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    prefer_cjk = contains_cjk(text)
    cjk_candidates = [
        fonts_dir / "yugothic.ttc",
        fonts_dir / "yugothib.ttf",
        fonts_dir / "meiryo.ttc",
        fonts_dir / "meiryo.ttf",
        fonts_dir / "msgothic.ttc",
        fonts_dir / "msmincho.ttc",
        fonts_dir / "msyh.ttc",
        fonts_dir / "simsun.ttc",
        fonts_dir / "simhei.ttf",
        fonts_dir / "malgun.ttf",
        fonts_dir / "arialuni.ttf",
    ]
    latin_candidates = [
        fonts_dir / "segoeui.ttf",
        fonts_dir / "arial.ttf",
        fonts_dir / "calibri.ttf",
    ]
    candidates = cjk_candidates + latin_candidates if prefer_cjk else latin_candidates + cjk_candidates
    for font_path in candidates:
        if font_path.exists():
            return font_path
    return None
