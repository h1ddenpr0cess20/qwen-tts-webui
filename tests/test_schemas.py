"""Tests for Pydantic request schemas (app/schemas.py).

Validates that schema defaults, required fields, constraints, and enum
literals all behave correctly, and that invalid payloads are rejected.
"""

import pytest
from pydantic import ValidationError

from app.schemas import ConvertRequest, TTSRequest, VideoRequest, VoiceProfileCreate


class TestTTSRequest:
    """TTSRequest validation tests."""

    def test_minimal_valid_request(self):
        req = TTSRequest(text="Hello world")
        assert req.text == "Hello world"
        assert req.mode == "custom_voice"
        assert req.chunk_text is True
        assert req.language is None
        assert req.model_id is None
        assert req.speaker is None
        assert req.instruct is None
        assert req.ref_audio is None
        assert req.ref_text is None
        assert req.x_vector_only_mode is False
        assert req.voice_profile is None

    def test_all_modes_accepted(self):
        for mode in ("custom_voice", "voice_design", "voice_clone"):
            req = TTSRequest(text="test", mode=mode)
            assert req.mode == mode

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            TTSRequest(text="test", mode="invalid_mode")
        errors = exc_info.value.errors()
        assert any("mode" in str(e.get("loc", "")) for e in errors)

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            TTSRequest(text="")
        errors = exc_info.value.errors()
        assert any("text" in str(e.get("loc", "")) for e in errors)

    def test_missing_text_rejected(self):
        with pytest.raises(ValidationError):
            TTSRequest()

    def test_voice_clone_fields_populated(self):
        req = TTSRequest(
            text="hello",
            mode="voice_clone",
            ref_audio="data:audio/wav;base64,abc",
            ref_text="transcript",
            x_vector_only_mode=True,
            voice_profile="myprofile",
        )
        assert req.ref_audio == "data:audio/wav;base64,abc"
        assert req.ref_text == "transcript"
        assert req.x_vector_only_mode is True
        assert req.voice_profile == "myprofile"

    def test_chunk_text_defaults_true(self):
        req = TTSRequest(text="hello")
        assert req.chunk_text is True

    def test_chunk_text_can_be_disabled(self):
        req = TTSRequest(text="hello", chunk_text=False)
        assert req.chunk_text is False


class TestConvertRequest:
    """ConvertRequest validation tests."""

    def test_valid_convert_request(self):
        req = ConvertRequest(data_url="data:audio/wav;base64,AAAA")
        assert req.target_format == "mp3"
        assert req.data_url == "data:audio/wav;base64,AAAA"

    def test_missing_data_url_rejected(self):
        with pytest.raises(ValidationError):
            ConvertRequest()

    def test_invalid_target_format_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            ConvertRequest(data_url="data:audio/wav;base64,AAAA", target_format="ogg")
        errors = exc_info.value.errors()
        assert any("target_format" in str(e.get("loc", "")) for e in errors)

    def test_default_format_is_mp3(self):
        req = ConvertRequest(data_url="x")
        assert req.target_format == "mp3"


class TestVideoRequest:
    """VideoRequest validation tests."""

    def test_valid_minimal_request(self):
        req = VideoRequest(data_url="data:audio/wav;base64,AAAA")
        assert req.style == "waveform"
        assert req.layout == "vertical"
        assert req.transcript is None

    def test_all_styles_accepted(self):
        for style in ("waveform", "spectrum", "pulse"):
            req = VideoRequest(data_url="x", style=style)
            assert req.style == style

    def test_all_layouts_accepted(self):
        for layout in ("vertical", "square", "landscape"):
            req = VideoRequest(data_url="x", layout=layout)
            assert req.layout == layout

    def test_invalid_style_rejected(self):
        with pytest.raises(ValidationError):
            VideoRequest(data_url="x", style="neon")

    def test_invalid_layout_rejected(self):
        with pytest.raises(ValidationError):
            VideoRequest(data_url="x", layout="4k")

    def test_transcript_optional(self):
        req = VideoRequest(data_url="x", transcript="Hello world")
        assert req.transcript == "Hello world"


class TestVoiceProfileCreate:
    """VoiceProfileCreate validation tests."""

    def test_valid_profile_creation(self):
        req = VoiceProfileCreate(name="myvoice", ref_audio="http://example.com/audio.wav")
        assert req.name == "myvoice"
        assert req.ref_audio == "http://example.com/audio.wav"
        assert req.ref_text is None
        assert req.x_vector_only_mode is False
        assert req.model_id is None
        assert req.device is None

    def test_name_min_length_enforced(self):
        with pytest.raises(ValidationError) as exc_info:
            VoiceProfileCreate(name="", ref_audio="http://example.com/audio.wav")
        errors = exc_info.value.errors()
        assert any("name" in str(e.get("loc", "")) for e in errors)

    def test_name_max_length_enforced(self):
        with pytest.raises(ValidationError) as exc_info:
            VoiceProfileCreate(name="a" * 65, ref_audio="http://example.com/audio.wav")
        errors = exc_info.value.errors()
        assert any("name" in str(e.get("loc", "")) for e in errors)

    def test_name_at_max_length_accepted(self):
        req = VoiceProfileCreate(name="a" * 64, ref_audio="x")
        assert len(req.name) == 64

    def test_missing_ref_audio_rejected(self):
        with pytest.raises(ValidationError):
            VoiceProfileCreate(name="voice")

    def test_all_optional_fields(self):
        req = VoiceProfileCreate(
            name="v",
            ref_audio="x",
            ref_text="hello",
            x_vector_only_mode=True,
            model_id="custom/model",
            device="cuda:1",
        )
        assert req.ref_text == "hello"
        assert req.x_vector_only_mode is True
        assert req.model_id == "custom/model"
        assert req.device == "cuda:1"
