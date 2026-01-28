"""Text-to-speech synthesis services."""

import io
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.config import (
    DEFAULT_CLONE_MODEL,
    DEFAULT_CUSTOM_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_VOICE_DESIGN_MODEL,
)
from app.schemas import TTSRequest
from app.services.model_loader import load_model
from app.services.voice_profiles import load_voice_profile
from app.utils.audio import prepare_ref_audio
from app.utils.text import split_text_chunks


def resolve_model_id(payload: TTSRequest, profile_meta: Optional[Dict[str, object]]) -> str:
    """Resolve the model ID based on request mode and profile metadata.

    Args:
        payload: TTS request payload.
        profile_meta: Optional voice profile metadata.

    Returns:
        Model identifier to load.
    """

    if payload.mode == "voice_clone" and payload.voice_profile:
        profile_model = (profile_meta or {}).get("model_id") or DEFAULT_CLONE_MODEL
        if payload.model_id and payload.model_id != profile_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Voice profile '{payload.voice_profile}' was saved with model '{profile_model}'. "
                    f"Use that model or re-save the profile for '{payload.model_id}'."
                ),
            )
        return profile_model
    if payload.model_id:
        return payload.model_id
    if payload.mode == "custom_voice":
        return DEFAULT_CUSTOM_MODEL
    if payload.mode == "voice_design":
        return DEFAULT_VOICE_DESIGN_MODEL
    if payload.mode == "voice_clone":
        return DEFAULT_CLONE_MODEL
    return DEFAULT_CUSTOM_MODEL


def synthesize_tts(payload: TTSRequest) -> StreamingResponse:
    """Synthesize speech from text according to the payload.

    Args:
        payload: TTS request payload.

    Returns:
        StreamingResponse with WAV audio.

    Raises:
        HTTPException: If synthesis fails or inputs are invalid.
    """

    profile_meta = None
    if payload.mode == "voice_clone" and payload.voice_profile:
        profile_meta = load_voice_profile(payload.voice_profile)

    model_id = resolve_model_id(payload, profile_meta)

    if not payload.chunk_text and len(payload.text) > 500:
        raise HTTPException(
            status_code=400,
            detail="Text exceeds 500 characters. Shorten it or enable chunk_text to auto-split.",
        )

    text_chunks = (
        split_text_chunks(payload.text, limit=500) if payload.chunk_text else [payload.text]
    )

    device = payload.device or DEFAULT_DEVICE
    model = load_model(model_id, device)

    sr_global = None
    audio_parts = []

    if payload.mode == "custom_voice":
        speaker = payload.speaker
        if speaker is None:
            supported = model.get_supported_speakers()
            if not supported:
                raise HTTPException(status_code=400, detail="No speakers available for this model.")
            speaker = supported[0]
        for chunk in text_chunks:
            wavs, sr = model.generate_custom_voice(
                text=chunk,
                language=payload.language or "Auto",
                speaker=speaker,
                instruct=payload.instruct or "",
            )
            if sr_global is None:
                sr_global = sr
            elif sr_global != sr:
                raise HTTPException(status_code=500, detail="Sample rate mismatch between chunks.")
            audio_parts.append(wavs[0])
    elif payload.mode == "voice_design":
        if not payload.instruct:
            raise HTTPException(
                status_code=422,
                detail="Voice design mode requires 'instruct' to describe the target voice.",
            )
        for chunk in text_chunks:
            wavs, sr = model.generate_voice_design(
                text=chunk,
                language=payload.language or "Auto",
                instruct=payload.instruct,
            )
            if sr_global is None:
                sr_global = sr
            elif sr_global != sr:
                raise HTTPException(status_code=500, detail="Sample rate mismatch between chunks.")
            audio_parts.append(wavs[0])
    elif payload.mode == "voice_clone":
        if payload.voice_profile:
            profile_data = profile_meta or load_voice_profile(payload.voice_profile)
            for chunk in text_chunks:
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=payload.language or "Auto",
                    voice_clone_prompt=profile_data["prompt_items"],
                )
                if sr_global is None:
                    sr_global = sr
                elif sr_global != sr:
                    raise HTTPException(status_code=500, detail="Sample rate mismatch between chunks.")
                audio_parts.append(wavs[0])
        else:
            if not payload.ref_audio:
                raise HTTPException(
                    status_code=422,
                    detail="Voice clone mode requires 'ref_audio' (URL, local path, or base64).",
                )
            if not payload.x_vector_only_mode and not payload.ref_text:
                raise HTTPException(
                    status_code=422,
                    detail="Voice clone mode needs 'ref_text' unless x_vector_only_mode is true.",
                )
            ref_audio_input = prepare_ref_audio(payload.ref_audio)
            for chunk in text_chunks:
                wavs, sr = model.generate_voice_clone(
                    text=chunk,
                    language=payload.language or "Auto",
                    ref_audio=ref_audio_input,
                    ref_text=payload.ref_text or "",
                    x_vector_only_mode=payload.x_vector_only_mode,
                )
                if sr_global is None:
                    sr_global = sr
                elif sr_global != sr:
                    raise HTTPException(status_code=500, detail="Sample rate mismatch between chunks.")
                audio_parts.append(wavs[0])
    else:
        raise HTTPException(status_code=400, detail="Unsupported mode.")

    if not audio_parts:
        raise HTTPException(status_code=500, detail="No audio generated.")
    combined_audio = np.concatenate(audio_parts, axis=0)

    buffer = io.BytesIO()
    sf.write(buffer, combined_audio, sr_global or 24000, format="WAV")
    buffer.seek(0)

    filename = f"qwen3_tts_{payload.mode}.wav"
    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )
