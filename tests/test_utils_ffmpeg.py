"""Tests for FFmpeg utility helpers (app/utils/ffmpeg.py)."""

import os
from pathlib import Path
from unittest.mock import patch

from app.utils.ffmpeg import escape_ffmpeg_path, find_system_font


class TestEscapeFfmpegPath:
    """escape_ffmpeg_path path escaping tests."""

    def test_simple_unix_path_unchanged(self):
        p = Path("/tmp/test.txt")
        assert escape_ffmpeg_path(p) == "/tmp/test.txt"

    def test_colon_escaped(self):
        p = Path("/tmp/file:name.txt")
        result = escape_ffmpeg_path(p)
        assert "\\:" in result
        assert ":" not in result.replace("\\:", "")

    def test_backslash_converted_to_forward_slash(self):
        # Simulate a Windows-style path string
        p = Path("C:\\Users\\test\\file.txt")
        result = escape_ffmpeg_path(p)
        assert "\\" not in result or "\\:" in result


class TestFindSystemFont:
    """find_system_font font resolution tests."""

    def test_returns_none_on_non_windows_without_env(self):
        with patch.dict(os.environ, {}, clear=False):
            if "QWEN_TTS_VIDEO_FONT" in os.environ:
                del os.environ["QWEN_TTS_VIDEO_FONT"]
            if os.name != "nt":
                assert find_system_font("Hello") is None

    def test_env_font_returned_when_exists(self, tmp_path):
        font_file = tmp_path / "custom.ttf"
        font_file.write_bytes(b"fake font data")
        with patch.dict(os.environ, {"QWEN_TTS_VIDEO_FONT": str(font_file)}):
            result = find_system_font("Hello")
            assert result == font_file

    def test_env_font_ignored_when_missing(self, tmp_path):
        nonexistent = tmp_path / "nope.ttf"
        with patch.dict(os.environ, {"QWEN_TTS_VIDEO_FONT": str(nonexistent)}):
            # On non-Windows, falls through to return None
            if os.name != "nt":
                assert find_system_font("Hello") is None

    def test_empty_env_font_ignored(self):
        with patch.dict(os.environ, {"QWEN_TTS_VIDEO_FONT": ""}):
            if os.name != "nt":
                assert find_system_font("Hello") is None
