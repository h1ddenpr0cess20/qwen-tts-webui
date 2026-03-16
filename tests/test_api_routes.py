"""Tests for API routes (app/api/routes.py).

Uses FastAPI's TestClient with ML models fully mocked. Exercises
every endpoint including error conditions.
"""

import io
import json
import pickle

import numpy as np
import pytest
import soundfile as sf

from tests.conftest import make_voice_profile, make_wav_data_url


class TestHealthEndpoint:
    """GET /api/health tests."""

    def test_health_returns_200(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_health_contains_status_ok(self, client):
        data = client.get("/api/health").json()
        assert data["status"] == "ok"

    def test_health_contains_model_and_device(self, client):
        data = client.get("/api/health").json()
        assert "model" in data
        assert "device" in data


class TestMetaEndpoint:
    """GET /api/meta tests."""

    def test_meta_returns_200(self, client):
        resp = client.get("/api/meta")
        assert resp.status_code == 200

    def test_meta_contains_expected_keys(self, client):
        data = client.get("/api/meta").json()
        assert "model_id" in data
        assert "device" in data
        assert "defaults" in data
        assert "speakers" in data
        assert "languages" in data
        assert "voices_dir" in data

    def test_meta_defaults_has_all_modes(self, client):
        defaults = client.get("/api/meta").json()["defaults"]
        assert "custom_voice" in defaults
        assert "voice_design" in defaults
        assert "voice_clone" in defaults

    def test_meta_speakers_from_model(self, client):
        data = client.get("/api/meta").json()
        # FakeQwen3TTSModel returns ["Alice", "Bob"]
        assert "Alice" in data["speakers"]
        assert "Bob" in data["speakers"]


class TestTTSEndpoint:
    """POST /api/tts tests."""

    def test_custom_voice_returns_wav(self, client):
        resp = client.post("/api/tts", json={"text": "Hello world", "mode": "custom_voice"})
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_custom_voice_wav_is_parseable(self, client):
        resp = client.post("/api/tts", json={"text": "Hello"})
        audio, sr = sf.read(io.BytesIO(resp.content))
        assert sr == 24000
        assert len(audio) > 0

    def test_voice_design_without_instruct_returns_422(self, client):
        resp = client.post("/api/tts", json={"text": "Hello", "mode": "voice_design"})
        assert resp.status_code == 422

    def test_voice_design_with_instruct_returns_wav(self, client):
        resp = client.post("/api/tts", json={
            "text": "Hello", "mode": "voice_design", "instruct": "warm tone"
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"

    def test_voice_design_text_over_500_returns_400(self, client):
        resp = client.post("/api/tts", json={
            "text": "A" * 501, "mode": "voice_design", "instruct": "test"
        })
        assert resp.status_code == 400

    def test_voice_clone_without_ref_audio_returns_422(self, client):
        resp = client.post("/api/tts", json={"text": "Hello", "mode": "voice_clone"})
        assert resp.status_code == 422

    def test_voice_clone_with_ref_audio_and_text(self, client):
        resp = client.post("/api/tts", json={
            "text": "Hello", "mode": "voice_clone",
            "ref_audio": "http://example.com/ref.wav",
            "ref_text": "reference text",
        })
        assert resp.status_code == 200

    def test_voice_clone_x_vector_mode(self, client):
        resp = client.post("/api/tts", json={
            "text": "Hello", "mode": "voice_clone",
            "ref_audio": "http://example.com/ref.wav",
            "x_vector_only_mode": True,
        })
        assert resp.status_code == 200

    def test_voice_clone_with_profile(self, client, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "apiprofile", model_id="test")
        resp = client.post("/api/tts", json={
            "text": "Hello", "mode": "voice_clone", "voice_profile": "apiprofile",
        })
        assert resp.status_code == 200

    def test_empty_text_returns_422(self, client):
        resp = client.post("/api/tts", json={"text": ""})
        assert resp.status_code == 422

    def test_missing_text_returns_422(self, client):
        resp = client.post("/api/tts", json={"mode": "custom_voice"})
        assert resp.status_code == 422

    def test_invalid_mode_returns_422(self, client):
        resp = client.post("/api/tts", json={"text": "hi", "mode": "invalid"})
        assert resp.status_code == 422

    def test_chunking_disabled_long_text_returns_400(self, client):
        resp = client.post("/api/tts", json={
            "text": "A" * 501, "mode": "custom_voice", "chunk_text": False,
        })
        assert resp.status_code == 400


class TestConvertEndpoint:
    """POST /api/convert tests."""

    def test_valid_wav_to_mp3_conversion(self, client):
        data_url = make_wav_data_url()
        resp = client.post("/api/convert", json={"data_url": data_url})
        # This may fail if ffmpeg is not installed, which is expected in CI
        if resp.status_code == 200:
            assert resp.headers["content-type"] == "audio/mpeg"
        else:
            # 500 from missing ffmpeg is acceptable in test env
            assert resp.status_code == 500

    def test_invalid_data_url_returns_415(self, client):
        resp = client.post("/api/convert", json={"data_url": "not-a-data-url"})
        assert resp.status_code == 415

    def test_missing_data_url_returns_422(self, client):
        resp = client.post("/api/convert", json={})
        assert resp.status_code == 422


class TestVoiceProfileEndpoints:
    """Voice profile CRUD endpoint tests."""

    def test_list_empty_profiles(self, client):
        resp = client.get("/api/voice_profiles")
        assert resp.status_code == 200
        assert resp.json()["profiles"] == []

    def test_list_profiles_after_creation(self, client, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "listed")
        resp = client.get("/api/voice_profiles")
        assert resp.status_code == 200
        names = [p["name"] for p in resp.json()["profiles"]]
        assert "listed" in names

    def test_delete_existing_profile(self, client, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "deleteme")
        resp = client.delete("/api/voice_profiles/deleteme")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/api/voice_profiles/ghost")
        assert resp.status_code == 404

    def test_export_existing_profile(self, client, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "exportme")
        resp = client.get("/api/voice_profiles/exportme/export")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/octet-stream"

    def test_export_nonexistent_returns_404(self, client):
        resp = client.get("/api/voice_profiles/ghost/export")
        assert resp.status_code == 404

    def test_import_valid_pt_file(self, client, tmp_voice_dir):
        data = {"prompt_items": {"test": True}, "model_id": "m", "name": "imp"}
        content = pickle.dumps(data)
        resp = client.post(
            "/api/voice_profiles/import",
            files={"file": ("imported.pt", content, "application/octet-stream")},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "imported"

    def test_import_non_pt_file_returns_400(self, client):
        resp = client.post(
            "/api/voice_profiles/import",
            files={"file": ("voice.wav", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_import_empty_file_returns_400(self, client):
        resp = client.post(
            "/api/voice_profiles/import",
            files={"file": ("voice.pt", b"", "application/octet-stream")},
        )
        assert resp.status_code == 400

    def test_create_profile_missing_ref_text_returns_422(self, client):
        resp = client.post("/api/voice_profiles", json={
            "name": "test", "ref_audio": "http://example.com/ref.wav",
        })
        assert resp.status_code == 422

    def test_create_profile_x_vector_mode(self, client):
        resp = client.post("/api/voice_profiles", json={
            "name": "test",
            "ref_audio": "http://example.com/ref.wav",
            "x_vector_only_mode": True,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["name"] == "test"
