"""HTTP API routes for the TTS service."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from app.config import (
    DEFAULT_CLONE_MODEL,
    DEFAULT_CUSTOM_MODEL,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_ID,
    DEFAULT_VOICE_DESIGN_MODEL,
    VOICE_PROFILE_DIR,
)
from app.schemas import ConvertRequest, TTSRequest, VideoRequest, VoiceProfileCreate
from app.services.audio_conversion import convert_wav_data_url_to_mp3
from app.services.model_loader import load_model
from app.services.tts_synthesizer import synthesize_tts
from app.services.video_renderer import render_video
from app.services.voice_profiles import (
    create_voice_profile as create_voice_profile_service,
    delete_voice_profile,
    import_voice_profile_file,
    list_voice_profiles,
    normalize_profile_name,
    resolve_profile_device,
    resolve_profile_model_id,
    sanitize_name,
)

router = APIRouter()


@router.post("/api/convert")
def convert_audio(payload: ConvertRequest) -> StreamingResponse:
    """Convert WAV audio to MP3.

    Args:
        payload: Conversion request payload.

    Returns:
        StreamingResponse containing MP3 audio.
    """

    if payload.target_format != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 conversion is supported.")
    return convert_wav_data_url_to_mp3(payload.data_url)


@router.post("/api/video")
def render_video_endpoint(payload: VideoRequest) -> FileResponse:
    """Render a visualization video for the provided audio.

    Args:
        payload: Video render request payload.

    Returns:
        FileResponse containing the generated MP4 video.
    """

    return render_video(payload.data_url, payload.transcript, payload.style, payload.layout)


@router.get("/api/meta")
def meta() -> dict:
    """Return metadata about available models and configuration.

    Returns:
        Dict containing model defaults and supported speakers/languages.
    """

    model = load_model(DEFAULT_CUSTOM_MODEL, DEFAULT_DEVICE)
    try:
        speakers = model.get_supported_speakers()
    except Exception:
        speakers = []
    try:
        languages = model.get_supported_languages()
    except Exception:
        languages = []
    return {
        "model_id": DEFAULT_CUSTOM_MODEL,
        "device": DEFAULT_DEVICE,
        "defaults": {
            "custom_voice": DEFAULT_CUSTOM_MODEL,
            "voice_design": DEFAULT_VOICE_DESIGN_MODEL,
            "voice_clone": DEFAULT_CLONE_MODEL,
        },
        "speakers": speakers,
        "languages": languages,
        "voices_dir": str(VOICE_PROFILE_DIR),
    }


@router.get("/api/voice_profiles")
def list_voice_profiles_endpoint() -> dict:
    """List saved voice profiles.

    Returns:
        Dict containing saved voice profile metadata.
    """

    return {"profiles": list_voice_profiles()}


@router.get("/api/voice_profiles/{name}/export")
def export_voice_profile(name: str) -> FileResponse:
    """Export a saved voice profile file.

    Args:
        name: Profile name to export.

    Returns:
        FileResponse containing the voice profile file.
    """

    safe = normalize_profile_name(name)
    pt_path = VOICE_PROFILE_DIR / f"{safe}.pt"
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    return FileResponse(
        pt_path,
        media_type="application/octet-stream",
        filename=f"{safe}.pt",
    )


@router.post("/api/voice_profiles")
def create_voice_profile(payload: VoiceProfileCreate) -> dict:
    """Create a new voice profile from reference audio.

    Args:
        payload: Voice profile creation payload.

    Returns:
        Dict describing the created profile.
    """

    model_id = resolve_profile_model_id(payload)
    device = resolve_profile_device(payload)
    return create_voice_profile_service(payload, model_id, device)


@router.post("/api/voice_profiles/import")
async def import_voice_profile(file: UploadFile = File(...)) -> dict:
    """Import a voice profile .pt file."""

    content = await file.read()
    return import_voice_profile_file(file.filename, content)


@router.delete("/api/voice_profiles/{name}")
def delete_voice_profile_endpoint(name: str) -> dict:
    """Delete a stored voice profile.

    Args:
        name: Profile name to delete.

    Returns:
        Dict acknowledging deletion.
    """

    delete_voice_profile(name)
    return {"status": "deleted", "name": sanitize_name(name)}


@router.post("/api/tts")
def synthesize(payload: TTSRequest) -> StreamingResponse:
    """Synthesize speech audio from text.

    Args:
        payload: TTS request payload.

    Returns:
        StreamingResponse containing WAV audio.
    """

    return synthesize_tts(payload)


@router.get("/api/health")
def health() -> dict:
    """Return a basic health check payload.

    Returns:
        Dict with service status and defaults.
    """

    return {"status": "ok", "model": DEFAULT_MODEL_ID, "device": DEFAULT_DEVICE}
