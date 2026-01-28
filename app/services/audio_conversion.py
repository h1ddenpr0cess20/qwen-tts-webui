"""Audio conversion services."""

import io

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.utils.audio import data_url_to_wav_buffer


def convert_wav_data_url_to_mp3(data_url: str) -> StreamingResponse:
    """Convert a WAV data URL to an MP3 streaming response.

    Args:
        data_url: Data URL containing WAV audio.

    Returns:
        StreamingResponse containing MP3 audio.

    Raises:
        HTTPException: If conversion fails or dependencies are missing.
    """

    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="MP3 conversion unavailable: install pydub and ffmpeg.",
        ) from exc

    try:
        wav_buf = data_url_to_wav_buffer(data_url)
        mp3_buf = io.BytesIO()
        AudioSegment.from_file(wav_buf, format="wav").export(mp3_buf, format="mp3", bitrate="192k")
        mp3_buf.seek(0)
        return StreamingResponse(
            mp3_buf,
            media_type="audio/mpeg",
            headers={"Content-Disposition": 'inline; filename="qwen3_tts.mp3"'},
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to convert audio. Ensure ffmpeg/avconv is installed.",
        ) from exc
