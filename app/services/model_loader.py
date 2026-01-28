"""Model loading utilities with caching."""

from functools import lru_cache

from app.config import DEFAULT_DEVICE, DEFAULT_MODEL_ID, USE_FLASH
from app.deps import Qwen3TTSModel, torch


def resolve_dtype(device: str) -> torch.dtype:
    """Resolve the torch dtype based on the target device.

    Args:
        device: Device identifier (e.g., "cpu", "cuda:0").

    Returns:
        torch.dtype appropriate for the device.
    """

    if device.startswith("cuda") or device.startswith("mps"):
        return torch.bfloat16
    return torch.float32


@lru_cache(maxsize=3)
def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = DEFAULT_DEVICE) -> Qwen3TTSModel:
    """Load and cache a Qwen3 TTS model.

    Args:
        model_id: Model identifier or local path.
        device: Device mapping string for model loading.

    Returns:
        Loaded Qwen3 TTS model instance.
    """

    dtype = resolve_dtype(device)
    kwargs = {"device_map": device, "dtype": dtype}
    if USE_FLASH:
        kwargs["attn_implementation"] = "flash_attention_2"
    return Qwen3TTSModel.from_pretrained(model_id, **kwargs)
