"""Video rendering services built on FFmpeg."""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.config import VIDEO_LAYOUTS
from app.utils.audio import audio_duration_seconds, decode_data_url
from app.utils.ffmpeg import escape_ffmpeg_path, find_system_font
from app.utils.files import cleanup_files
from app.utils.text import contains_cjk, wrap_transcript


def render_video(payload_data_url: str, transcript: Optional[str], style: str, layout: str) -> FileResponse:
    """Render a video visualization from audio and optional transcript.

    Args:
        payload_data_url: Audio data URL.
        transcript: Optional transcript text to overlay.
        style: Visualization style (waveform, spectrum, pulse).
        layout: Output layout (vertical, square, landscape).

    Returns:
        FileResponse with the rendered MP4 video.

    Raises:
        HTTPException: If rendering fails or inputs are invalid.
    """

    audio_bytes = decode_data_url(payload_data_url)
    if audio_bytes is None:
        raise HTTPException(status_code=415, detail="Invalid audio data URL.")

    chosen_layout = layout or "vertical"
    if chosen_layout not in VIDEO_LAYOUTS:
        raise HTTPException(status_code=400, detail="Unsupported layout.")
    width, height = VIDEO_LAYOUTS[chosen_layout]
    fps = 30

    raw_transcript = transcript or ""
    duration_sec = audio_duration_seconds(audio_bytes)

    audio_path: Optional[Path] = None
    text_path: Optional[Path] = None
    output_path: Optional[Path] = None
    rendered = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_file.write(audio_bytes)
            audio_path = Path(audio_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_file:
            output_path = Path(out_file.name)

        bg_color = "0x0b1020"
        accent = "0x00c2a8"
        if style == "spectrum":
            if duration_sec and duration_sec > 0:
                spec_width = int(max(160, min(width * 4, duration_sec * fps)))
            else:
                spec_width = width
            viz_filter = (
                f"[1:a]showspectrum=s={spec_width}x{height}:mode=combined:color=rainbow:scale=log:fps={fps},"
                f"format=rgba,scale={width}:{height},setsar=1,colorkey=0x000000:0.02:0.0 [viz]"
            )
        elif style == "pulse":
            viz_filter = (
                f"[1:a]aformat=channel_layouts=stereo,adelay=0|12,"
                f"avectorscope=s={width}x{height}:mode=polar:draw=aaline:scale=log:zoom=1.35"
                f":rc=0:gc=194:bc=168,"
                f"format=rgba,boxblur=2:1,"
                f"scale={width}:{height},setsar=1,"
                "colorkey=0x000000:0.08:0.0 [viz]"
            )
        else:
            viz_filter = (
                f"[1:a]showwaves=s={width}x{height}:mode=line:rate={fps}:colors={accent},"
                "format=rgba,setsar=1,colorkey=0x000000:0.12:0.0 [viz]"
            )

        drawtext_filter = ""
        if raw_transcript.strip():
            cjk = contains_cjk(raw_transcript)
            scale_factor = (
                0.038 if chosen_layout == "vertical" else 0.035 if chosen_layout == "square" else 0.032
            )
            font_size = max(32, min(72, int(height * scale_factor)))
            max_lines = 5 if chosen_layout == "vertical" else 4 if chosen_layout == "square" else 4
            glyph_ratio = 0.9 if cjk else 0.55
            max_width = max(18, int(width / (font_size * glyph_ratio)))
            transcript_text = wrap_transcript(
                raw_transcript,
                max_width,
                max_lines=max_lines,
                break_long_words=cjk,
            )
            if transcript_text:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as text_file:
                    text_file.write(transcript_text)
                    text_path = Path(text_file.name)
                line_spacing = int(font_size * 0.25)
                center_ratio = (
                    0.62 if chosen_layout == "vertical" else 0.60 if chosen_layout == "square" else 0.58
                )
                text_y = f"(h*{center_ratio})-(text_h/2)"
                escaped_text = escape_ffmpeg_path(text_path)
                fontfile = find_system_font(raw_transcript)
                font_part = ""
                if fontfile:
                    font_part = f":fontfile='{escape_ffmpeg_path(fontfile)}'"
                drawtext_filter = (
                    f"drawtext=textfile='{escaped_text}'{font_part}:fontcolor=white:fontsize={font_size}"
                    f":line_spacing={line_spacing}:x=(w-text_w)/2:y={text_y}"
                    f":shadowcolor=0x000000@0.6:shadowx=3:shadowy=3"
                )

        if drawtext_filter:
            filter_complex = (
                f"{viz_filter};[0:v][viz]overlay=0:0[base];[base]{drawtext_filter} [v]"
            )
        else:
            filter_complex = f"{viz_filter};[0:v][viz]overlay=0:0 [v]"

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"color=c={bg_color}:s={width}x{height}:r={fps}",
            "-i",
            str(audio_path),
            "-filter_complex",
            filter_complex,
            "-map",
            "[v]",
            "-map",
            "1:a",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-r",
            str(fps),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(output_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        rendered = True
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        detail = "Video render failed. Ensure ffmpeg is installed and supports drawtext."
        if stderr:
            snippet = stderr.replace("\r", " ").replace("\n", " ")
            detail = f"{detail} ffmpeg error: {snippet[:240]}"
        raise HTTPException(
            status_code=500,
            detail=detail,
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail="Video render failed. ffmpeg was not found on PATH.",
        ) from exc
    finally:
        if not rendered:
            cleanup_files(audio_path, text_path, output_path)

    background = BackgroundTask(cleanup_files, audio_path, text_path, output_path)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="qwen3_tts.mp4",
        background=background,
    )
