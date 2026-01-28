"""Pydantic request schemas for the API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    """Request payload for TTS synthesis.

    Attributes:
        mode: Synthesis mode (custom voice, voice design, or voice clone).
        text: Text to synthesize.
        chunk_text: Whether to split text into 500-char chunks.
        language: Optional language tag (auto when omitted).
        model_id: Optional model ID or local path override.
        device: Optional device map for loading the model.
        speaker: Speaker name for custom voice mode.
        instruct: Style guidance; required for voice design mode.
        ref_audio: Reference audio for voice cloning.
        ref_text: Transcript for reference audio in voice clone mode.
        x_vector_only_mode: Use speaker embedding only when true.
        voice_profile: Saved voice profile name for voice clone mode.
    """

    mode: Literal["custom_voice", "voice_design", "voice_clone"] = "custom_voice"
    text: str = Field(..., min_length=1, description="Text to synthesize")
    chunk_text: bool = Field(
        default=True,
        description="Auto-split text into <=500 char chunks to avoid model limit.",
    )
    language: Optional[str] = Field(
        default=None, description="Language tag; leave empty for auto"
    )
    model_id: Optional[str] = Field(default=None, description="Model id or local path")
    device: Optional[str] = Field(default=None, description="Device map for loading")
    speaker: Optional[str] = Field(
        default=None, description="Speaker name for custom_voice mode"
    )
    instruct: Optional[str] = Field(
        default=None,
        description="Style guidance; required for voice_design mode, optional otherwise",
    )
    ref_audio: Optional[str] = Field(
        default=None,
        description="Reference audio for voice_clone. Can be a URL, local path, or base64 data URI.",
    )
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of ref_audio for voice_clone. Optional when x_vector_only_mode is true.",
    )
    x_vector_only_mode: bool = Field(
        default=False,
        description="When true, only uses speaker embedding; skips ref_text. May reduce quality.",
    )
    voice_profile: Optional[str] = Field(
        default=None, description="Saved voice profile name (uses cached prompt)."
    )


class ConvertRequest(BaseModel):
    """Request payload for audio conversion.

    Attributes:
        target_format: Target audio format (mp3 only).
        data_url: data URL containing WAV audio to convert.
    """

    target_format: Literal["mp3"] = "mp3"
    data_url: str = Field(..., description="data URL (audio/wav) to convert")


class VideoRequest(BaseModel):
    """Request payload for video rendering.

    Attributes:
        data_url: data URL containing WAV audio to render.
        transcript: Optional transcript to overlay.
        style: Visualization style.
        layout: Output layout aspect ratio.
    """

    data_url: str = Field(..., description="data URL (audio/wav) to render")
    transcript: Optional[str] = Field(default=None, description="Transcript text to overlay")
    style: Literal["waveform", "spectrum", "pulse"] = "waveform"
    layout: Literal["vertical", "square", "landscape"] = "vertical"


class VoiceProfileCreate(BaseModel):
    """Request payload for creating a voice profile.

    Attributes:
        name: Profile name.
        ref_audio: Reference audio (URL, local path, or data URL).
        ref_text: Transcript for reference audio.
        x_vector_only_mode: Use speaker embedding only when true.
        model_id: Optional model ID override for cloning.
        device: Optional device override.
    """

    name: str = Field(..., min_length=1, max_length=64, description="Profile name")
    ref_audio: str = Field(..., description="Reference audio (URL/local/data URL)")
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of ref_audio. Optional when x_vector_only_mode is true.",
    )
    x_vector_only_mode: bool = False
    model_id: Optional[str] = None
    device: Optional[str] = None
