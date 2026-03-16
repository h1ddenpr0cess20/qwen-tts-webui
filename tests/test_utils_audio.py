"""Tests for audio utility functions (app/utils/audio.py).

Tests decode_data_url, data_url_to_wav_buffer, audio_duration_seconds,
and prepare_ref_audio with real WAV data where possible.
"""

import base64
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from app.utils.audio import (
    audio_duration_seconds,
    data_url_to_wav_buffer,
    decode_data_url,
    prepare_ref_audio,
)


def _make_wav_bytes(duration_samples: int = 4800, sr: int = 24000) -> bytes:
    t = np.linspace(0, duration_samples / sr, duration_samples, dtype=np.float32)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    return buf.getvalue()


def _make_data_url(wav_bytes: bytes) -> str:
    b64 = base64.b64encode(wav_bytes).decode()
    return f"data:audio/wav;base64,{b64}"


class TestDecodeDataUrl:
    """decode_data_url parsing and base64 decoding tests."""

    def test_valid_wav_data_url(self):
        wav = _make_wav_bytes()
        url = _make_data_url(wav)
        result = decode_data_url(url)
        assert result == wav

    def test_returns_none_for_non_audio_prefix(self):
        b64 = base64.b64encode(b"hello").decode()
        assert decode_data_url(f"data:text/plain;base64,{b64}") is None

    def test_returns_none_for_missing_base64_marker(self):
        assert decode_data_url("data:audio/wav,AAAA") is None

    def test_returns_none_for_empty_string(self):
        assert decode_data_url("") is None

    def test_returns_none_for_non_string(self):
        assert decode_data_url(42) is None  # type: ignore[arg-type]

    def test_returns_none_for_invalid_base64(self):
        assert decode_data_url("data:audio/wav;base64,!!!not-base64!!!") is None

    def test_various_audio_subtypes(self):
        raw = b"fakewav"
        b64 = base64.b64encode(raw).decode()
        for subtype in ("wav", "ogg", "flac", "mpeg"):
            url = f"data:audio/{subtype};base64,{b64}"
            assert decode_data_url(url) == raw

    def test_url_without_data_prefix(self):
        assert decode_data_url("http://example.com/audio.wav") is None


class TestDataUrlToWavBuffer:
    """data_url_to_wav_buffer conversion tests."""

    def test_returns_seekable_buffer_at_start(self):
        wav = _make_wav_bytes()
        url = _make_data_url(wav)
        buf = data_url_to_wav_buffer(url)
        assert buf.tell() == 0
        assert buf.read() == wav

    def test_invalid_url_raises_415(self):
        with pytest.raises(HTTPException) as exc_info:
            data_url_to_wav_buffer("not-a-data-url")
        assert exc_info.value.status_code == 415

    def test_buffer_is_readable_by_soundfile(self):
        wav = _make_wav_bytes(sr=16000)
        url = _make_data_url(wav)
        buf = data_url_to_wav_buffer(url)
        audio_data, sr = sf.read(buf)
        assert sr == 16000
        assert len(audio_data) > 0


class TestAudioDurationSeconds:
    """audio_duration_seconds estimation tests."""

    def test_correct_duration_for_known_wav(self):
        sr = 24000
        samples = sr * 2  # exactly 2 seconds
        wav = _make_wav_bytes(duration_samples=samples, sr=sr)
        duration = audio_duration_seconds(wav)
        assert duration is not None
        assert abs(duration - 2.0) < 0.01

    def test_returns_none_for_garbage_bytes(self):
        assert audio_duration_seconds(b"this is not audio") is None

    def test_returns_none_for_empty_bytes(self):
        assert audio_duration_seconds(b"") is None

    def test_short_audio_duration(self):
        sr = 48000
        samples = 480  # 0.01 seconds
        wav = _make_wav_bytes(duration_samples=samples, sr=sr)
        duration = audio_duration_seconds(wav)
        assert duration is not None
        assert abs(duration - 0.01) < 0.001


class TestPrepareRefAudio:
    """prepare_ref_audio data-URL vs path pass-through tests."""

    def test_non_data_url_passed_through_as_string(self):
        result = prepare_ref_audio("http://example.com/audio.wav")
        assert result == "http://example.com/audio.wav"

    def test_local_path_passed_through(self):
        result = prepare_ref_audio("/tmp/audio.wav")
        assert result == "/tmp/audio.wav"

    def test_valid_wav_data_url_decoded_to_tuple(self):
        wav = _make_wav_bytes(duration_samples=4800, sr=24000)
        url = _make_data_url(wav)
        result = prepare_ref_audio(url)
        assert isinstance(result, tuple)
        audio_np, sr = result
        assert isinstance(audio_np, np.ndarray)
        assert sr == 24000
        assert audio_np.ndim == 1

    def test_stereo_wav_downmixed_to_mono(self):
        sr = 24000
        samples = 2400
        stereo = np.random.randn(samples, 2).astype(np.float32)
        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV")
        b64 = base64.b64encode(buf.getvalue()).decode()
        url = f"data:audio/wav;base64,{b64}"
        result = prepare_ref_audio(url)
        assert isinstance(result, tuple)
        audio_np, sr_out = result
        assert audio_np.ndim == 1
        assert sr_out == sr

    def test_invalid_data_url_raises_415(self):
        with pytest.raises(HTTPException) as exc_info:
            prepare_ref_audio("data:audio/wav;base64,!!!invalid!!!")
        assert exc_info.value.status_code == 415
