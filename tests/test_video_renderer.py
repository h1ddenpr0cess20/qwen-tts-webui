"""Tests for video renderer service (app/services/video_renderer.py).

Since ffmpeg subprocess calls are involved, these tests mock subprocess.run
to verify the command construction and error handling without needing
ffmpeg installed.
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

from app.services.video_renderer import render_video
from tests.conftest import make_wav_data_url


class TestRenderVideo:
    """render_video request handling and error tests."""

    def test_invalid_data_url_raises_415(self):
        with pytest.raises(HTTPException) as exc_info:
            render_video("not-a-data-url", None, "waveform", "vertical")
        assert exc_info.value.status_code == 415

    def test_unsupported_layout_raises_400(self):
        data_url = make_wav_data_url()
        with pytest.raises(HTTPException) as exc_info:
            render_video(data_url, None, "waveform", "ultrawide")
        assert exc_info.value.status_code == 400
        assert "layout" in exc_info.value.detail.lower()

    def test_ffmpeg_not_found_raises_500(self):
        data_url = make_wav_data_url()
        with patch("app.services.video_renderer.subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(HTTPException) as exc_info:
                render_video(data_url, None, "waveform", "vertical")
            assert exc_info.value.status_code == 500
            assert "not found" in exc_info.value.detail.lower()

    def test_ffmpeg_failure_raises_500_with_stderr(self):
        data_url = make_wav_data_url()
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr="Filter parsing failed")
        with patch("app.services.video_renderer.subprocess.run", side_effect=error):
            with pytest.raises(HTTPException) as exc_info:
                render_video(data_url, None, "waveform", "vertical")
            assert exc_info.value.status_code == 500
            assert "Filter parsing failed" in exc_info.value.detail

    @patch("app.services.video_renderer.subprocess.run")
    def test_successful_render_returns_file_response(self, mock_run, tmp_path):
        """Simulate a successful ffmpeg render."""
        data_url = make_wav_data_url()

        def fake_run(cmd, **kwargs):
            # Find the output path (last argument to ffmpeg)
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake mp4 data")

        mock_run.side_effect = fake_run
        response = render_video(data_url, None, "waveform", "vertical")
        assert response.media_type == "video/mp4"

    @patch("app.services.video_renderer.subprocess.run")
    def test_all_styles_produce_valid_commands(self, mock_run):
        """Verify all three visualization styles produce a command."""
        data_url = make_wav_data_url()

        for style in ("waveform", "spectrum", "pulse"):
            mock_run.reset_mock()

            def fake_run(cmd, **kwargs):
                output_path = Path(cmd[-1])
                output_path.write_bytes(b"fake mp4")

            mock_run.side_effect = fake_run
            response = render_video(data_url, None, style, "vertical")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "ffmpeg"
            assert "-filter_complex" in cmd

    @patch("app.services.video_renderer.subprocess.run")
    def test_all_layouts_use_correct_dimensions(self, mock_run):
        """Verify each layout uses the right width/height in ffmpeg command."""
        from app.config import VIDEO_LAYOUTS
        data_url = make_wav_data_url()

        for layout_name, (w, h) in VIDEO_LAYOUTS.items():
            mock_run.reset_mock()

            def fake_run(cmd, **kwargs):
                output_path = Path(cmd[-1])
                output_path.write_bytes(b"fake mp4")

            mock_run.side_effect = fake_run
            render_video(data_url, None, "waveform", layout_name)
            cmd = mock_run.call_args[0][0]
            cmd_str = " ".join(cmd)
            assert f"s={w}x{h}" in cmd_str

    @patch("app.services.video_renderer.subprocess.run")
    def test_transcript_included_in_filter(self, mock_run):
        data_url = make_wav_data_url()

        def fake_run(cmd, **kwargs):
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake mp4")

        mock_run.side_effect = fake_run
        render_video(data_url, "Hello world transcript", "waveform", "vertical")
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "drawtext" in cmd_str

    @patch("app.services.video_renderer.subprocess.run")
    def test_empty_transcript_no_drawtext(self, mock_run):
        data_url = make_wav_data_url()

        def fake_run(cmd, **kwargs):
            output_path = Path(cmd[-1])
            output_path.write_bytes(b"fake mp4")

        mock_run.side_effect = fake_run
        render_video(data_url, "", "waveform", "vertical")
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "drawtext" not in cmd_str

    def test_temp_files_cleaned_on_ffmpeg_failure(self):
        data_url = make_wav_data_url()
        error = subprocess.CalledProcessError(1, "ffmpeg", stderr="error")
        with patch("app.services.video_renderer.subprocess.run", side_effect=error):
            with patch("app.services.video_renderer.cleanup_files") as mock_cleanup:
                with pytest.raises(HTTPException):
                    render_video(data_url, None, "waveform", "vertical")
                mock_cleanup.assert_called_once()
                # All three path args should be passed
                args = mock_cleanup.call_args[0]
                assert len(args) == 3
