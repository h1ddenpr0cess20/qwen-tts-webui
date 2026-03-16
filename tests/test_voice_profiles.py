"""Tests for voice profile service (app/services/voice_profiles.py).

These tests exercise the profile CRUD helpers using real filesystem
operations on a temp directory. The torch save/load are replaced by
pickle in conftest.py.
"""

import json
import pickle
import time

import pytest
from fastapi import HTTPException

from app.services.voice_profiles import (
    delete_voice_profile,
    import_voice_profile_file,
    list_voice_profiles,
    load_voice_profile,
    normalize_profile_name,
    profile_paths,
    resolve_profile_device,
    resolve_profile_model_id,
    sanitize_name,
    save_voice_profile,
)
from app.schemas import VoiceProfileCreate
from tests.conftest import make_voice_profile


class TestSanitizeName:
    """sanitize_name input cleaning tests."""

    def test_alphanumeric_unchanged(self):
        assert sanitize_name("MyVoice123") == "MyVoice123"

    def test_special_chars_replaced_with_underscore(self):
        # Trailing underscores are stripped, so "my voice!@#$%" -> "my_voice"
        result = sanitize_name("my voice!@#$%")
        assert " " not in result
        assert "!" not in result
        assert result == "my_voice"

    def test_dots_hyphens_underscores_preserved(self):
        assert sanitize_name("my-voice_v2.0") == "my-voice_v2.0"

    def test_leading_trailing_underscores_stripped(self):
        assert sanitize_name("___name___") == "name"

    def test_empty_after_sanitization_raises_400(self):
        with pytest.raises(HTTPException) as exc_info:
            sanitize_name("!!!!")
        assert exc_info.value.status_code == 400

    def test_single_valid_char(self):
        assert sanitize_name("a") == "a"

    def test_spaces_replaced(self):
        result = sanitize_name("my voice")
        assert " " not in result


class TestNormalizeProfileName:
    """normalize_profile_name .pt suffix handling tests."""

    def test_strips_pt_suffix(self):
        assert normalize_profile_name("voice.pt") == "voice"

    def test_strips_pt_case_insensitive(self):
        assert normalize_profile_name("voice.PT") == "voice"

    def test_no_suffix_passes_through(self):
        assert normalize_profile_name("voice") == "voice"

    def test_double_pt_strips_only_last(self):
        assert normalize_profile_name("voice.pt.pt") == "voice.pt"


class TestProfilePaths:
    """profile_paths path generation tests."""

    def test_returns_pt_and_meta_paths(self, tmp_voice_dir):
        pt, meta = profile_paths("testvoice")
        assert pt.name == "testvoice.pt"
        assert meta.name == "testvoice.meta.json"
        assert pt.parent == tmp_voice_dir
        assert meta.parent == tmp_voice_dir


class TestSaveAndLoadVoiceProfile:
    """save_voice_profile and load_voice_profile round-trip tests."""

    def test_save_creates_pt_and_meta_files(self, tmp_voice_dir):
        save_voice_profile("testv", {"prompt": "data"}, "model-1", original_name="Test V")
        pt_path = tmp_voice_dir / "testv.pt"
        meta_path = tmp_voice_dir / "testv.meta.json"
        assert pt_path.exists()
        assert meta_path.exists()

    def test_meta_json_contains_expected_keys(self, tmp_voice_dir):
        save_voice_profile("testv", {"prompt": "data"}, "model-1")
        meta_path = tmp_voice_dir / "testv.meta.json"
        meta = json.loads(meta_path.read_text())
        assert meta["name"] == "testv"
        assert meta["model_id"] == "model-1"
        assert "saved_at" in meta

    def test_load_recovers_saved_data(self, tmp_voice_dir):
        prompt = {"fake_prompt": True, "ref_text": "hello"}
        save_voice_profile("roundtrip", prompt, "model-x", original_name="Round Trip")
        loaded = load_voice_profile("roundtrip")
        assert loaded["prompt_items"] == prompt
        assert loaded["model_id"] == "model-x"
        assert loaded["original_name"] == "Round Trip"

    def test_load_missing_profile_raises_404(self, tmp_voice_dir):
        with pytest.raises(HTTPException) as exc_info:
            load_voice_profile("nonexistent")
        assert exc_info.value.status_code == 404

    def test_load_corrupted_file_raises_500(self, tmp_voice_dir):
        pt_path = tmp_voice_dir / "bad.pt"
        pt_path.write_bytes(b"not a valid pickle")
        with pytest.raises(HTTPException) as exc_info:
            load_voice_profile("bad")
        assert exc_info.value.status_code == 500

    def test_load_file_without_prompt_items_raises_500(self, tmp_voice_dir):
        pt_path = tmp_voice_dir / "noprompt.pt"
        with open(pt_path, "wb") as f:
            pickle.dump({"model_id": "x"}, f)  # missing prompt_items
        with pytest.raises(HTTPException) as exc_info:
            load_voice_profile("noprompt")
        assert exc_info.value.status_code == 500
        assert "Corrupted" in exc_info.value.detail


class TestDeleteVoiceProfile:
    """delete_voice_profile removal tests."""

    def test_delete_removes_both_files(self, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "todelete")
        delete_voice_profile("todelete")
        assert not (tmp_voice_dir / "todelete.pt").exists()
        assert not (tmp_voice_dir / "todelete.meta.json").exists()

    def test_delete_nonexistent_raises_404(self, tmp_voice_dir):
        with pytest.raises(HTTPException) as exc_info:
            delete_voice_profile("ghost")
        assert exc_info.value.status_code == 404

    def test_delete_with_only_pt_file(self, tmp_voice_dir):
        pt_path = tmp_voice_dir / "ptonly.pt"
        with open(pt_path, "wb") as f:
            pickle.dump({"prompt_items": {}}, f)
        delete_voice_profile("ptonly")
        assert not pt_path.exists()


class TestListVoiceProfiles:
    """list_voice_profiles enumeration tests."""

    def test_empty_dir_returns_empty_list(self, tmp_voice_dir):
        result = list_voice_profiles()
        assert result == []

    def test_lists_profiles_with_metadata(self, tmp_voice_dir):
        make_voice_profile(tmp_voice_dir, "alice", model_id="model-a")
        make_voice_profile(tmp_voice_dir, "bob", model_id="model-b")
        result = list_voice_profiles()
        names = [p["name"] for p in result]
        assert "alice" in names
        assert "bob" in names

    def test_sorted_by_saved_at_descending(self, tmp_voice_dir):
        data_old = {
            "prompt_items": {}, "model_id": "m", "saved_at": 1000.0,
            "name": "old", "original_name": "old",
        }
        data_new = {
            "prompt_items": {}, "model_id": "m", "saved_at": 2000.0,
            "name": "new", "original_name": "new",
        }
        for name, data in [("old", data_old), ("new", data_new)]:
            pt = tmp_voice_dir / f"{name}.pt"
            meta = tmp_voice_dir / f"{name}.meta.json"
            with open(pt, "wb") as f:
                pickle.dump(data, f)
            meta.write_text(json.dumps({
                "name": name, "original_name": name,
                "model_id": "m", "saved_at": data["saved_at"],
            }))

        result = list_voice_profiles()
        assert result[0]["name"] == "new"
        assert result[1]["name"] == "old"

    def test_handles_corrupted_pt_gracefully(self, tmp_voice_dir):
        bad_pt = tmp_voice_dir / "corrupt.pt"
        bad_pt.write_bytes(b"garbage")
        result = list_voice_profiles()
        assert len(result) == 1
        assert result[0]["name"] == "corrupt"
        assert result[0]["model_id"] is None

    def test_falls_back_to_pt_data_when_no_meta_json(self, tmp_voice_dir):
        data = {
            "prompt_items": {}, "model_id": "fallback-model",
            "saved_at": 1234.0, "name": "nojson",
        }
        pt = tmp_voice_dir / "nojson.pt"
        with open(pt, "wb") as f:
            pickle.dump(data, f)
        # No .meta.json file
        result = list_voice_profiles()
        assert len(result) == 1
        assert result[0]["name"] == "nojson"
        assert result[0]["model_id"] == "fallback-model"


class TestImportVoiceProfileFile:
    """import_voice_profile_file upload validation tests."""

    def test_missing_filename_raises_400(self, tmp_voice_dir):
        with pytest.raises(HTTPException) as exc_info:
            import_voice_profile_file("", b"data")
        assert exc_info.value.status_code == 400

    def test_non_pt_filename_raises_400(self, tmp_voice_dir):
        with pytest.raises(HTTPException) as exc_info:
            import_voice_profile_file("voice.wav", b"data")
        assert exc_info.value.status_code == 400

    def test_empty_content_raises_400(self, tmp_voice_dir):
        with pytest.raises(HTTPException) as exc_info:
            import_voice_profile_file("voice.pt", b"")
        assert exc_info.value.status_code == 400

    def test_invalid_pickle_raises_400(self, tmp_voice_dir):
        with pytest.raises(HTTPException) as exc_info:
            import_voice_profile_file("voice.pt", b"not a valid pickle")
        assert exc_info.value.status_code == 400

    def test_valid_import_creates_files(self, tmp_voice_dir):
        data = {"prompt_items": {"test": True}, "model_id": "m", "saved_at": 100.0, "name": "imp"}
        content = pickle.dumps(data)
        result = import_voice_profile_file("imported.pt", content)
        assert result["status"] == "imported"
        assert result["name"] == "imported"
        assert (tmp_voice_dir / "imported.pt").exists()
        assert (tmp_voice_dir / "imported.meta.json").exists()

    def test_missing_prompt_items_raises_400(self, tmp_voice_dir):
        data = {"model_id": "m"}
        content = pickle.dumps(data)
        with pytest.raises(HTTPException) as exc_info:
            import_voice_profile_file("bad.pt", content)
        assert exc_info.value.status_code == 400
        assert "Corrupted" in exc_info.value.detail


class TestResolveProfileModelIdAndDevice:
    """resolve_profile_model_id and resolve_profile_device tests."""

    def test_uses_payload_model_id_when_provided(self):
        payload = VoiceProfileCreate(name="v", ref_audio="x", model_id="custom/model")
        assert resolve_profile_model_id(payload) == "custom/model"

    def test_falls_back_to_default_clone_model(self):
        payload = VoiceProfileCreate(name="v", ref_audio="x")
        from app.config import DEFAULT_CLONE_MODEL
        assert resolve_profile_model_id(payload) == DEFAULT_CLONE_MODEL

    def test_uses_payload_device_when_provided(self):
        payload = VoiceProfileCreate(name="v", ref_audio="x", device="cuda:1")
        assert resolve_profile_device(payload) == "cuda:1"

    def test_falls_back_to_default_device(self):
        payload = VoiceProfileCreate(name="v", ref_audio="x")
        from app.config import DEFAULT_DEVICE
        assert resolve_profile_device(payload) == DEFAULT_DEVICE
