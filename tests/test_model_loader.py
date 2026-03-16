"""Tests for model loader (app/services/model_loader.py).

The actual model loading is mocked, but we test resolve_dtype logic
and verify that load_model passes the right arguments.
"""

from unittest.mock import patch, MagicMock

from app.services.model_loader import resolve_dtype, load_model


class TestResolveDtype:
    """resolve_dtype device-to-dtype mapping tests."""

    def test_cuda_device_returns_bfloat16(self):
        import app.deps
        result = resolve_dtype("cuda:0")
        assert result == app.deps.torch.bfloat16

    def test_cuda_no_index_returns_bfloat16(self):
        import app.deps
        assert resolve_dtype("cuda") == app.deps.torch.bfloat16

    def test_mps_device_returns_bfloat16(self):
        import app.deps
        assert resolve_dtype("mps") == app.deps.torch.bfloat16

    def test_cpu_device_returns_float32(self):
        import app.deps
        assert resolve_dtype("cpu") == app.deps.torch.float32

    def test_unknown_device_returns_float32(self):
        import app.deps
        assert resolve_dtype("xpu:0") == app.deps.torch.float32


class TestLoadModel:
    """load_model caching and argument forwarding tests."""

    def test_load_model_calls_from_pretrained(self):
        # Clear the lru_cache to get a clean call
        load_model.cache_clear()
        from tests.conftest import FakeQwen3TTSModel
        with patch.object(FakeQwen3TTSModel, "from_pretrained", wraps=FakeQwen3TTSModel.from_pretrained) as spy:
            with patch("app.services.model_loader.Qwen3TTSModel", FakeQwen3TTSModel):
                model = load_model("test-model-1", "cpu")
                spy.assert_called_once()
                call_kwargs = spy.call_args
                assert call_kwargs[0][0] == "test-model-1"
                assert call_kwargs[1]["device_map"] == "cpu"
        load_model.cache_clear()

    def test_load_model_caches_result(self):
        load_model.cache_clear()
        from tests.conftest import FakeQwen3TTSModel
        with patch("app.services.model_loader.Qwen3TTSModel", FakeQwen3TTSModel):
            m1 = load_model("cached-model", "cpu")
            m2 = load_model("cached-model", "cpu")
            assert m1 is m2
        load_model.cache_clear()

    def test_different_args_get_different_cache_entries(self):
        load_model.cache_clear()
        from tests.conftest import FakeQwen3TTSModel
        with patch("app.services.model_loader.Qwen3TTSModel", FakeQwen3TTSModel):
            m1 = load_model("model-a", "cpu")
            m2 = load_model("model-b", "cpu")
            assert m1 is not m2
        load_model.cache_clear()
