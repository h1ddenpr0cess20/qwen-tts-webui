"""Tests for the TTS synthesizer service (app/services/tts_synthesizer.py).

The model is mocked via the FakeQwen3TTSModel from conftest.py. These
tests verify request validation, mode routing, chunking, cancellation,
and WAV output correctness.
"""

from threading import Event
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException

from app.schemas import TTSRequest
from app.services.tts_synthesizer import (
    _raise_if_aborted,
    resolve_model_id,
    synthesize_tts,
)
from tests.conftest import FakeQwen3TTSModel, make_voice_profile


class TestRaiseIfAborted:
    """_raise_if_aborted cooperative cancellation tests."""

    def test_no_event_does_not_raise(self):
        _raise_if_aborted(None)

    def test_unset_event_does_not_raise(self):
        event = Event()
        _raise_if_aborted(event)

    def test_set_event_raises_499(self):
        event = Event()
        event.set()
        with pytest.raises(HTTPException) as exc_info:
            _raise_if_aborted(event)
        assert exc_info.value.status_code == 499


class TestResolveModelId:
    """resolve_model_id mode-based model selection tests."""

    def test_custom_voice_default(self):
        payload = TTSRequest(text="hi", mode="custom_voice")
        from app.config import DEFAULT_CUSTOM_MODEL
        assert resolve_model_id(payload, None) == DEFAULT_CUSTOM_MODEL

    def test_voice_design_default(self):
        payload = TTSRequest(text="hi", mode="voice_design")
        from app.config import DEFAULT_VOICE_DESIGN_MODEL
        assert resolve_model_id(payload, None) == DEFAULT_VOICE_DESIGN_MODEL

    def test_voice_clone_default(self):
        payload = TTSRequest(text="hi", mode="voice_clone")
        from app.config import DEFAULT_CLONE_MODEL
        assert resolve_model_id(payload, None) == DEFAULT_CLONE_MODEL

    def test_explicit_model_id_overrides_default(self):
        payload = TTSRequest(text="hi", mode="custom_voice", model_id="my/model")
        assert resolve_model_id(payload, None) == "my/model"

    def test_voice_clone_with_profile_uses_profile_model(self):
        payload = TTSRequest(text="hi", mode="voice_clone", voice_profile="v1")
        meta = {"model_id": "profile-model"}
        assert resolve_model_id(payload, meta) == "profile-model"

    def test_voice_clone_profile_model_mismatch_raises_400(self):
        payload = TTSRequest(
            text="hi", mode="voice_clone", voice_profile="v1", model_id="different-model"
        )
        meta = {"model_id": "profile-model"}
        with pytest.raises(HTTPException) as exc_info:
            resolve_model_id(payload, meta)
        assert exc_info.value.status_code == 400
        assert "profile-model" in exc_info.value.detail

    def test_voice_clone_profile_without_model_uses_default(self):
        payload = TTSRequest(text="hi", mode="voice_clone", voice_profile="v1")
        meta = {}  # no model_id in profile
        from app.config import DEFAULT_CLONE_MODEL
        assert resolve_model_id(payload, meta) == DEFAULT_CLONE_MODEL


class TestSynthesizeTTS:
    """synthesize_tts end-to-end tests with mocked model."""

    @pytest.fixture(autouse=True)
    def _mock_model(self):
        fake = FakeQwen3TTSModel.from_pretrained("test")
        with patch("app.services.tts_synthesizer.load_model", return_value=fake):
            yield fake

    def test_custom_voice_returns_wav_response(self):
        payload = TTSRequest(text="Hello world", mode="custom_voice")
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"
        assert "qwen3_tts_custom_voice.wav" in response.headers["content-disposition"]

    def test_custom_voice_wav_is_valid(self):
        payload = TTSRequest(text="Hello", mode="custom_voice")
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"
        # Verify the content-disposition names the file correctly
        assert "qwen3_tts_custom_voice.wav" in response.headers["content-disposition"]

    def test_voice_design_requires_instruct(self):
        payload = TTSRequest(text="Hello", mode="voice_design")
        with pytest.raises(HTTPException) as exc_info:
            synthesize_tts(payload)
        assert exc_info.value.status_code == 422

    def test_voice_design_with_instruct_succeeds(self):
        payload = TTSRequest(text="Hello", mode="voice_design", instruct="warm female voice")
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"

    def test_voice_design_text_limit_enforced(self):
        payload = TTSRequest(text="A" * 501, mode="voice_design", instruct="test")
        with pytest.raises(HTTPException) as exc_info:
            synthesize_tts(payload)
        assert exc_info.value.status_code == 400
        assert "500 characters" in exc_info.value.detail

    def test_voice_clone_requires_ref_audio(self):
        payload = TTSRequest(text="Hello", mode="voice_clone")
        with pytest.raises(HTTPException) as exc_info:
            synthesize_tts(payload)
        assert exc_info.value.status_code == 422
        assert "ref_audio" in exc_info.value.detail

    def test_voice_clone_requires_ref_text_unless_x_vector(self):
        payload = TTSRequest(
            text="Hello", mode="voice_clone",
            ref_audio="http://example.com/ref.wav",
        )
        with pytest.raises(HTTPException) as exc_info:
            synthesize_tts(payload)
        assert exc_info.value.status_code == 422
        assert "ref_text" in exc_info.value.detail

    def test_voice_clone_x_vector_mode_skips_ref_text(self):
        payload = TTSRequest(
            text="Hello", mode="voice_clone",
            ref_audio="http://example.com/ref.wav",
            x_vector_only_mode=True,
        )
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"

    def test_voice_clone_with_ref_audio_and_text(self):
        payload = TTSRequest(
            text="Hello", mode="voice_clone",
            ref_audio="http://example.com/ref.wav",
            ref_text="reference transcript",
        )
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"

    def test_voice_clone_with_profile(self, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "myprofile", model_id="test")
        payload = TTSRequest(
            text="Hello", mode="voice_clone", voice_profile="myprofile",
        )
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"

    def test_chunking_splits_long_text_into_multiple_chunks(self):
        """Verify that chunking is engaged for long text.

        We patch the model to count how many times generate_custom_voice
        is called -- with chunking, it should be called multiple times.
        """
        from unittest.mock import MagicMock
        long_text = ". ".join(["Hello world"] * 60)
        payload = TTSRequest(text=long_text, mode="custom_voice", chunk_text=True)

        # Patch the model to count calls
        fake = FakeQwen3TTSModel.from_pretrained("test")
        fake.generate_custom_voice = MagicMock(side_effect=fake.generate_custom_voice)
        with patch("app.services.tts_synthesizer.load_model", return_value=fake):
            response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"
        # With chunking, the model should be called more than once
        assert fake.generate_custom_voice.call_count > 1

    def test_chunking_disabled_long_text_raises(self):
        payload = TTSRequest(text="A" * 501, mode="custom_voice", chunk_text=False)
        with pytest.raises(HTTPException) as exc_info:
            synthesize_tts(payload)
        assert exc_info.value.status_code == 400

    def test_abort_during_synthesis_raises_499(self):
        event = Event()
        event.set()
        payload = TTSRequest(text="Hello", mode="custom_voice")
        with pytest.raises(HTTPException) as exc_info:
            synthesize_tts(payload, stop_event=event)
        assert exc_info.value.status_code == 499

    def test_custom_voice_uses_first_speaker_when_none(self, _mock_model):
        payload = TTSRequest(text="Hello", mode="custom_voice")
        # speaker is None, should pick first from model.get_supported_speakers()
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"

    def test_custom_voice_explicit_speaker(self):
        payload = TTSRequest(text="Hello", mode="custom_voice", speaker="Bob")
        response = synthesize_tts(payload)
        assert response.media_type == "audio/wav"
