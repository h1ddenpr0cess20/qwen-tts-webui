"""Audio decoding and inspection helpers."""

from __future__ import annotations

import base64
import io
from typing import Optional, Tuple, Union

import numpy as np
import soundfile as sf
from fastapi import HTTPException

AudioInput = Union[str, Tuple[np.ndarray, int]]


def decode_data_url(data_url: str) -> Optional[bytes]:
    """Decode a base64-encoded audio data URL.

    Args:
        data_url: Data URL containing base64-encoded audio.

    Returns:
        The decoded bytes, or None if the payload is invalid.
    """

    if not isinstance(data_url, str):
        return None
    prefix = "data:audio/"
    marker = ";base64,"
    if not data_url.startswith(prefix) or marker not in data_url:
        return None
    try:
        payload = data_url.split(marker, 1)[1]
        return base64.b64decode(payload)
    except Exception:
        return None


def data_url_to_wav_buffer(data_url: str) -> io.BytesIO:
    """Convert a WAV data URL into a seekable buffer.

    Args:
        data_url: Data URL containing WAV audio.

    Returns:
        BytesIO buffer positioned at the start.

    Raises:
        HTTPException: If the data URL is invalid.
    """

    decoded = decode_data_url(data_url)
    if decoded is None:
        raise HTTPException(status_code=415, detail="Invalid data URL.")
    buf = io.BytesIO(decoded)
    buf.seek(0)
    return buf


def audio_duration_seconds(audio_bytes: bytes) -> Optional[float]:
    """Estimate audio duration from raw bytes.

    Args:
        audio_bytes: Raw audio file bytes.

    Returns:
        Duration in seconds when available, otherwise None.
    """

    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as audio_file:
            if audio_file.samplerate <= 0:
                return None
            return audio_file.frames / float(audio_file.samplerate)
    except Exception:
        return None


def prepare_ref_audio(ref_audio: str) -> AudioInput:
    """Prepare reference audio for voice cloning.

    Args:
        ref_audio: Reference audio input (URL/path or data URL).

    Returns:
        The original string when no decoding is needed, or (audio, sample_rate).

    Raises:
        HTTPException: If the data URL cannot be decoded.
    """

    ref_audio_input: AudioInput = ref_audio
    if isinstance(ref_audio_input, str) and ref_audio_input.startswith("data:audio/"):
        try:
            buf = data_url_to_wav_buffer(ref_audio_input)
            audio_np, sr_local = sf.read(buf, dtype="float32", always_2d=False)
            if audio_np.ndim > 1:
                audio_np = np.mean(audio_np, axis=-1).astype(np.float32)
            ref_audio_input = (audio_np, sr_local)
        except Exception as exc:
            raise HTTPException(
                status_code=415,
                detail="Could not decode recorded audio; please use WAV/OGG/FLAC.",
            ) from exc
    return ref_audio_input
