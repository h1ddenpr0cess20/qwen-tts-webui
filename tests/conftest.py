"""Shared test fixtures.

The heavy ML dependencies (torch, qwen_tts) are injected as mocks into
sys.modules BEFORE any app code is imported.  This lets the entire app
import chain resolve without needing GPU libraries installed.
"""

import io
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# 1. Fake torch module
# ---------------------------------------------------------------------------

_fake_torch = types.ModuleType("torch")
_fake_torch.float32 = "float32"
_fake_torch.bfloat16 = "bfloat16"


class _FakeDtype:
    pass


_fake_torch.dtype = _FakeDtype


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


_fake_torch.cuda = _FakeCuda()


def _fake_torch_save(data, path, **kwargs):
    """Persist profile data as JSON so tests can inspect it easily."""
    import pickle

    with open(path, "wb") as f:
        pickle.dump(data, f)


def _fake_torch_load(path, **kwargs):
    import pickle

    with open(path, "rb") as f:
        return pickle.load(f)


_fake_torch.save = _fake_torch_save
_fake_torch.load = _fake_torch_load

sys.modules["torch"] = _fake_torch
# Prevent real torch submodule imports from leaking in
for _sub in ("torch.utils", "torch.utils._pytree"):
    if _sub not in sys.modules:
        sys.modules[_sub] = types.ModuleType(_sub)

# ---------------------------------------------------------------------------
# 2. Fake qwen_tts module
# ---------------------------------------------------------------------------

_fake_qwen_tts = types.ModuleType("qwen_tts")


class FakeQwen3TTSModel:
    """Lightweight stand-in for the real TTS model.

    All generate_* methods return a deterministic short sine wave so that
    downstream code (WAV writing, concatenation) exercises real logic.
    """

    _speakers = ["Alice", "Bob"]
    _languages = ["en", "zh"]

    @classmethod
    def from_pretrained(cls, model_id, **kwargs):
        instance = cls()
        instance._model_id = model_id
        instance._load_kwargs = kwargs
        return instance

    def get_supported_speakers(self):
        return list(self._speakers)

    def get_supported_languages(self):
        return list(self._languages)

    @staticmethod
    def _make_audio(duration_samples: int = 2400, sr: int = 24000):
        t = np.linspace(0, duration_samples / sr, duration_samples, dtype=np.float32)
        wave = 0.5 * np.sin(2 * np.pi * 440 * t)
        return [wave], sr

    def generate_custom_voice(self, text, language, speaker, instruct=""):
        return self._make_audio()

    def generate_voice_design(self, text, language, instruct):
        return self._make_audio()

    def generate_voice_clone(self, text, language, **kwargs):
        return self._make_audio()

    def create_voice_clone_prompt(self, ref_audio, ref_text, x_vector_only_mode=False):
        return {"fake_prompt": True, "ref_text": ref_text}


_fake_qwen_tts.Qwen3TTSModel = FakeQwen3TTSModel
sys.modules["qwen_tts"] = _fake_qwen_tts

# ---------------------------------------------------------------------------
# 3. Patch app.deps so downstream imports resolve
# ---------------------------------------------------------------------------

_fake_deps = types.ModuleType("app.deps")
_fake_deps.torch = _fake_torch
_fake_deps.Qwen3TTSModel = FakeQwen3TTSModel
sys.modules["app.deps"] = _fake_deps

# Wire the fake deps module as an attribute of the app package so that
# `import app.deps` works after the app package is loaded.
import app as _app_pkg  # noqa: E402
_app_pkg.deps = _fake_deps  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_voice_dir(tmp_path, monkeypatch):
    """Point VOICE_PROFILE_DIR to a fresh temporary directory."""
    import app.config
    import app.services.voice_profiles
    import app.api.routes

    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    monkeypatch.setattr(app.config, "VOICE_PROFILE_DIR", voice_dir)
    monkeypatch.setattr(app.services.voice_profiles, "VOICE_PROFILE_DIR", voice_dir)
    monkeypatch.setattr(app.api.routes, "VOICE_PROFILE_DIR", voice_dir)
    return voice_dir


@pytest.fixture()
def client(tmp_voice_dir):
    """FastAPI TestClient with ML models mocked and a temp voice dir."""
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    with patch("app.services.model_loader.Qwen3TTSModel", FakeQwen3TTSModel):
        with patch("app.services.model_loader.load_model") as mock_load:
            mock_load.return_value = FakeQwen3TTSModel.from_pretrained("test-model")
            from app.main import app

            yield TestClient(app, raise_server_exceptions=False)


@pytest.fixture()
def fake_model():
    """Return a FakeQwen3TTSModel instance for direct service testing."""
    return FakeQwen3TTSModel.from_pretrained("test-model")


def make_wav_data_url(duration_samples: int = 2400, sr: int = 24000) -> str:
    """Build a valid WAV data URL from a sine wave."""
    import base64

    t = np.linspace(0, duration_samples / sr, duration_samples, dtype=np.float32)
    wave = 0.5 * np.sin(2 * np.pi * 440 * t)
    buf = io.BytesIO()
    sf.write(buf, wave, sr, format="WAV")
    raw = buf.getvalue()
    b64 = base64.b64encode(raw).decode()
    return f"data:audio/wav;base64,{b64}"


def make_voice_profile(voice_dir: Path, name: str = "testvoice", model_id: str = "test-model"):
    """Create a fake voice profile on disk and return its metadata."""
    import pickle
    import time

    data = {
        "prompt_items": {"fake_prompt": True},
        "model_id": model_id,
        "saved_at": time.time(),
        "name": name,
        "original_name": name,
    }
    pt_path = voice_dir / f"{name}.pt"
    meta_path = voice_dir / f"{name}.meta.json"
    with open(pt_path, "wb") as f:
        pickle.dump(data, f)
    meta = {k: data[k] for k in ("name", "original_name", "model_id", "saved_at")}
    meta_path.write_text(json.dumps(meta))
    return data
