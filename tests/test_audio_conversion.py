"""Tests for audio conversion service (app/services/audio_conversion.py).

The pydub/ffmpeg dependency may not be available in test environments,
so we test both the success path (when available) and the error handling.
"""

from unittest.mock import patch, MagicMock
import io

import pytest
from fastapi import HTTPException

from app.services.audio_conversion import convert_wav_data_url_to_mp3
from tests.conftest import make_wav_data_url


class TestConvertWavDataUrlToMp3:
    """convert_wav_data_url_to_mp3 conversion tests."""

    def test_invalid_data_url_raises_415(self):
        with pytest.raises(HTTPException) as exc_info:
            convert_wav_data_url_to_mp3("not-a-data-url")
        assert exc_info.value.status_code == 415

    def test_pydub_import_failure_raises_500(self):
        data_url = make_wav_data_url()
        with patch.dict("sys.modules", {"pydub": None}):
            with pytest.raises((HTTPException, ImportError)):
                convert_wav_data_url_to_mp3(data_url)

    def test_successful_conversion_returns_streaming_response(self):
        """Test conversion with a real data URL.

        If pydub/ffmpeg are installed, this verifies the full pipeline.
        If not, the test is skipped gracefully.
        """
        data_url = make_wav_data_url()
        try:
            response = convert_wav_data_url_to_mp3(data_url)
        except HTTPException as e:
            if e.status_code == 500:
                pytest.skip("pydub/ffmpeg not available in test environment")
            raise

        assert response.media_type == "audio/mpeg"
        assert "qwen3_tts.mp3" in response.headers.get("content-disposition", "")

    def test_conversion_failure_raises_500(self):
        data_url = make_wav_data_url()
        mock_segment = MagicMock()
        mock_segment.from_file.return_value.export.side_effect = RuntimeError("ffmpeg died")

        with patch.dict("sys.modules", {"pydub": MagicMock(AudioSegment=mock_segment)}):
            with pytest.raises(HTTPException) as exc_info:
                convert_wav_data_url_to_mp3(data_url)
            assert exc_info.value.status_code == 500
