import base64
import io
import json
import os
import re
import subprocess
import tempfile
import textwrap
import time
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

try:
    import torch
    from qwen_tts import Qwen3TTSModel
except Exception as exc:  # pragma: no cover - import guard for missing deps
    raise RuntimeError("qwen-tts and torch are required. Install from requirements.txt.") from exc


class TTSRequest(BaseModel):
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
    target_format: Literal["mp3"] = "mp3"
    data_url: str = Field(..., description="data URL (audio/wav) to convert")


class VideoRequest(BaseModel):
    data_url: str = Field(..., description="data URL (audio/wav) to render")
    transcript: Optional[str] = Field(default=None, description="Transcript text to overlay")
    style: Literal["waveform", "spectrum", "pulse"] = "waveform"
    layout: Literal["vertical", "square", "landscape"] = "vertical"


class VoiceProfileCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=64, description="Profile name")
    ref_audio: str = Field(..., description="Reference audio (URL/local/data URL)")
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of ref_audio. Optional when x_vector_only_mode is true.",
    )
    x_vector_only_mode: bool = False
    model_id: Optional[str] = None
    device: Optional[str] = None


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
VOICE_PROFILE_DIR = Path(os.getenv("VOICE_PROFILE_DIR", BASE_DIR / "voices"))
VOICE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_ID = os.getenv("QWEN_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
DEFAULT_DEVICE = os.getenv("QWEN_TTS_DEVICE")
if not DEFAULT_DEVICE:
    DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

DEFAULT_CUSTOM_MODEL = os.getenv("QWEN_TTS_CUSTOM_MODEL", DEFAULT_MODEL_ID)
DEFAULT_VOICE_DESIGN_MODEL = os.getenv(
    "QWEN_TTS_VD_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
)
DEFAULT_CLONE_MODEL = os.getenv("QWEN_TTS_CLONE_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")

USE_FLASH = os.getenv("QWEN_TTS_USE_FLASH", "").lower() in {"1", "true", "yes"}

VIDEO_LAYOUTS = {
    "vertical": (1080, 1920),
    "square": (1080, 1080),
    "landscape": (1920, 1080),
}

app = FastAPI(title="Qwen3 TTS Web API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _resolve_dtype(device: str):
    if device.startswith("cuda") or device.startswith("mps"):
        return torch.bfloat16
    return torch.float32


def _decode_data_url(data_url: str):
    match = re.match(r"data:audio/[^;]+;base64,(.+)", data_url)
    if not match:
        return None
    try:
        return base64.b64decode(match.group(1))
    except Exception:
        return None


def _escape_ffmpeg_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace(":", "\\:")


def _wrap_transcript(
    text: str,
    max_width: int,
    max_lines: int,
    break_long_words: bool = False,
) -> str:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    lines = textwrap.wrap(
        cleaned,
        width=max_width,
        break_long_words=break_long_words,
        break_on_hyphens=False,
    )
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        if len(lines[-1]) > max_width - 3:
            lines[-1] = lines[-1][: max_width - 3].rstrip()
        lines[-1] = lines[-1].rstrip(".") + "..."
    return "\n".join(lines)


def _contains_cjk(text: str) -> bool:
    for ch in text:
        code = ord(ch)
        if (
            0x3040 <= code <= 0x30FF  # Hiragana + Katakana
            or 0x31F0 <= code <= 0x31FF  # Katakana Phonetic Extensions
            or 0x4E00 <= code <= 0x9FFF  # CJK Unified Ideographs
            or 0x3400 <= code <= 0x4DBF  # CJK Extension A
            or 0x3000 <= code <= 0x303F  # CJK Symbols/Punctuation
            or 0xFF00 <= code <= 0xFFEF  # Fullwidth forms
            or 0x1100 <= code <= 0x11FF  # Hangul Jamo
            or 0xAC00 <= code <= 0xD7A3  # Hangul Syllables
        ):
            return True
    return False


def _find_system_font(text: str) -> Optional[Path]:
    env_font = os.getenv("QWEN_TTS_VIDEO_FONT", "").strip()
    if env_font:
        font_path = Path(env_font)
        if font_path.exists():
            return font_path
    if os.name != "nt":
        return None
    fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
    prefer_cjk = _contains_cjk(text)
    cjk_candidates = [
        fonts_dir / "yugothic.ttc",
        fonts_dir / "yugothib.ttf",
        fonts_dir / "meiryo.ttc",
        fonts_dir / "meiryo.ttf",
        fonts_dir / "msgothic.ttc",
        fonts_dir / "msmincho.ttc",
        fonts_dir / "msyh.ttc",
        fonts_dir / "simsun.ttc",
        fonts_dir / "simhei.ttf",
        fonts_dir / "malgun.ttf",
        fonts_dir / "arialuni.ttf",
    ]
    latin_candidates = [
        fonts_dir / "segoeui.ttf",
        fonts_dir / "arial.ttf",
        fonts_dir / "calibri.ttf",
    ]
    candidates = cjk_candidates + latin_candidates if prefer_cjk else latin_candidates + cjk_candidates
    for font_path in candidates:
        if font_path.exists():
            return font_path
    return None


def _cleanup_files(*paths: Optional[Path]):
    for path in paths:
        if not path:
            continue
        try:
            path.unlink()
        except FileNotFoundError:
            continue


def _split_text_chunks(text: str, limit: int = 500):
    """Split text into <=limit character chunks without breaking words."""
    if len(text) <= limit:
        return [text]
    tokens = re.findall(r"\S+\s*", text)
    chunks = []
    current = ""
    for tok in tokens:
        if len(tok) > limit:
            raise HTTPException(
                status_code=400,
                detail=f"A single word exceeds the {limit}-character limit: '{tok.strip()[:32]}'...",
            )
        if len(current) + len(tok) > limit:
            if current.strip():
                chunks.append(current.rstrip())
            current = tok
        else:
            current += tok
    if current.strip():
        chunks.append(current.rstrip())
    return chunks


def _data_url_to_wav_buffer(data_url: str) -> io.BytesIO:
    decoded = _decode_data_url(data_url)
    if decoded is None:
        raise HTTPException(status_code=415, detail="Invalid data URL.")
    buf = io.BytesIO(decoded)
    buf.seek(0)
    return buf


def _audio_duration_seconds(audio_bytes: bytes) -> Optional[float]:
    try:
        with sf.SoundFile(io.BytesIO(audio_bytes)) as audio_file:
            if audio_file.samplerate <= 0:
                return None
            return audio_file.frames / float(audio_file.samplerate)
    except Exception:
        return None


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_")
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid profile name.")
    return safe


def _normalize_profile_name(name: str) -> str:
    base = name[:-3] if name.lower().endswith(".pt") else name
    return _sanitize_name(base)


def _prepare_ref_audio(ref_audio: str):
    ref_audio_input = ref_audio
    if isinstance(ref_audio_input, str) and ref_audio_input.startswith("data:audio/"):
        try:
            buf = _data_url_to_wav_buffer(ref_audio_input)
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


def _profile_paths(name: str):
    safe = _normalize_profile_name(name)
    return VOICE_PROFILE_DIR / f"{safe}.pt", VOICE_PROFILE_DIR / f"{safe}.meta.json"


def _save_voice_profile(name: str, prompt_items, model_id: str, original_name: Optional[str] = None):
    safe = _sanitize_name(name)
    pt_path, meta_path = _profile_paths(safe)
    data = {
        "prompt_items": prompt_items,
        "model_id": model_id,
        "saved_at": time.time(),
        "name": safe,
        "original_name": original_name or name,
    }
    torch.save(data, pt_path)
    meta = {k: data[k] for k in ("name", "original_name", "model_id", "saved_at")}
    meta_path.write_text(json.dumps(meta))


def _load_voice_profile(name: str):
    pt_path, _ = _profile_paths(name)
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Failed to load voice profile. The file may be incompatible or corrupted.",
        ) from exc
    if "prompt_items" not in data:
        raise HTTPException(status_code=500, detail="Corrupted voice profile file.")
    return data


def _delete_voice_profile(name: str):
    pt_path, meta_path = _profile_paths(name)
    removed = False
    for path in (pt_path, meta_path):
        if path.exists():
            path.unlink()
            removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="Voice profile not found.")


def _list_voice_profiles():
    items = []
    for file in VOICE_PROFILE_DIR.glob("*.pt"):
        try:
            meta_path = file.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                items.append(
                    {
                        "name": meta.get("name") or file.stem,
                        "original_name": meta.get("original_name") or meta.get("name") or file.stem,
                        "model_id": meta.get("model_id"),
                        "saved_at": meta.get("saved_at"),
                    }
                )
                continue
            data = torch.load(file, map_location="cpu", weights_only=False)
            items.append(
                {
                    "name": data.get("name") or file.stem,
                    "original_name": data.get("original_name") or data.get("name") or file.stem,
                    "model_id": data.get("model_id"),
                    "saved_at": data.get("saved_at"),
                }
            )
        except Exception:
            items.append(
                {
                    "name": file.stem,
                    "original_name": file.stem,
                    "model_id": None,
                    "saved_at": None,
                }
            )
    return sorted(items, key=lambda x: x.get("saved_at") or 0, reverse=True)


@app.post("/api/convert")
def convert_audio(payload: ConvertRequest):
    if payload.target_format != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 conversion is supported.")
    try:
        from pydub import AudioSegment
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="MP3 conversion unavailable: install pydub and ffmpeg.",
        ) from exc

    try:
        wav_buf = _data_url_to_wav_buffer(payload.data_url)
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


@app.post("/api/video")
def render_video(payload: VideoRequest):
    audio_bytes = _decode_data_url(payload.data_url)
    if audio_bytes is None:
        raise HTTPException(status_code=415, detail="Invalid audio data URL.")

    layout = payload.layout or "vertical"
    if layout not in VIDEO_LAYOUTS:
        raise HTTPException(status_code=400, detail="Unsupported layout.")
    width, height = VIDEO_LAYOUTS[layout]
    fps = 30

    raw_transcript = payload.transcript or ""
    duration_sec = _audio_duration_seconds(audio_bytes)

    audio_path = None
    text_path = None
    output_path = None
    rendered = False
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
            audio_file.write(audio_bytes)
            audio_path = Path(audio_file.name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_file:
            output_path = Path(out_file.name)

        bg_color = "0x0b1020"
        accent = "0x00c2a8"
        if payload.style == "spectrum":
            if duration_sec and duration_sec > 0:
                spec_width = int(max(160, min(width * 4, duration_sec * fps)))
            else:
                spec_width = width
            viz_filter = (
                f"[1:a]showspectrum=s={spec_width}x{height}:mode=combined:color=rainbow:scale=log:fps={fps},"
                f"format=rgba,scale={width}:{height},setsar=1,colorkey=0x000000:0.02:0.0 [viz]"
            )
        elif payload.style == "pulse":
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
            cjk = _contains_cjk(raw_transcript)
            scale_factor = 0.038 if layout == "vertical" else 0.035 if layout == "square" else 0.032
            font_size = max(32, min(72, int(height * scale_factor)))
            max_lines = 5 if layout == "vertical" else 4 if layout == "square" else 4
            glyph_ratio = 0.9 if cjk else 0.55
            max_width = max(18, int(width / (font_size * glyph_ratio)))
            transcript = _wrap_transcript(
                raw_transcript,
                max_width,
                max_lines=max_lines,
                break_long_words=cjk,
            )
            if transcript:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".txt", mode="w", encoding="utf-8"
                ) as text_file:
                    text_file.write(transcript)
                    text_path = Path(text_file.name)
                line_spacing = int(font_size * 0.25)
                center_ratio = 0.62 if layout == "vertical" else 0.60 if layout == "square" else 0.58
                text_y = f"(h*{center_ratio})-(text_h/2)"
                escaped_text = _escape_ffmpeg_path(text_path)
                fontfile = _find_system_font(raw_transcript)
                font_part = ""
                if fontfile:
                    font_part = f":fontfile='{_escape_ffmpeg_path(fontfile)}'"
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
            _cleanup_files(audio_path, text_path, output_path)

    background = BackgroundTask(_cleanup_files, audio_path, text_path, output_path)
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename="qwen3_tts.mp4",
        background=background,
    )


@lru_cache(maxsize=3)
def load_model(model_id: str = DEFAULT_MODEL_ID, device: str = DEFAULT_DEVICE):
    dtype = _resolve_dtype(device)
    kwargs = {"device_map": device, "dtype": dtype}
    if USE_FLASH:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen3TTSModel.from_pretrained(model_id, **kwargs)
    return model


@app.get("/api/meta")
def meta():
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


@app.get("/api/voice_profiles")
def list_voice_profiles():
    return {"profiles": _list_voice_profiles()}


@app.get("/api/voice_profiles/{name}/export")
def export_voice_profile(name: str):
    safe = _normalize_profile_name(name)
    pt_path = VOICE_PROFILE_DIR / f"{safe}.pt"
    if not pt_path.exists():
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    return FileResponse(
        pt_path,
        media_type="application/octet-stream",
        filename=f"{safe}.pt",
    )


@app.post("/api/voice_profiles")
def create_voice_profile(payload: VoiceProfileCreate):
    model_id = payload.model_id or DEFAULT_CLONE_MODEL
    device = payload.device or DEFAULT_DEVICE
    model = load_model(model_id, device)

    if not payload.x_vector_only_mode and not payload.ref_text:
        raise HTTPException(
            status_code=422,
            detail="Voice profile needs 'ref_text' unless x_vector_only_mode is true.",
        )

    ref_audio_input = _prepare_ref_audio(payload.ref_audio)
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_input,
        ref_text=payload.ref_text or "",
        x_vector_only_mode=payload.x_vector_only_mode,
    )
    safe_name = _sanitize_name(payload.name)
    _save_voice_profile(safe_name, prompt_items, model_id, original_name=payload.name)
    return {
        "status": "ok",
        "name": safe_name,
        "original_name": payload.name,
        "model_id": model_id,
    }


@app.delete("/api/voice_profiles/{name}")
def delete_voice_profile(name: str):
    _delete_voice_profile(name)
    return {"status": "deleted", "name": _sanitize_name(name)}


@app.post("/api/tts")
def synthesize(payload: TTSRequest):
    profile_meta = None
    if payload.mode == "voice_clone" and payload.voice_profile:
        profile_meta = _load_voice_profile(payload.voice_profile)
        profile_model = profile_meta.get("model_id") or DEFAULT_CLONE_MODEL
        if payload.model_id and payload.model_id != profile_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Voice profile '{payload.voice_profile}' was saved with model '{profile_model}'. "
                    f"Use that model or re-save the profile for '{payload.model_id}'."
                ),
            )
        model_id = profile_model
    elif payload.model_id:
        model_id = payload.model_id
    else:
        if payload.mode == "custom_voice":
            model_id = DEFAULT_CUSTOM_MODEL
        elif payload.mode == "voice_design":
            model_id = DEFAULT_VOICE_DESIGN_MODEL
        elif payload.mode == "voice_clone":
            model_id = DEFAULT_CLONE_MODEL
        else:
            model_id = DEFAULT_CUSTOM_MODEL

    if not payload.chunk_text and len(payload.text) > 500:
        raise HTTPException(
            status_code=400,
            detail="Text exceeds 500 characters. Shorten it or enable chunk_text to auto-split.",
        )

    text_chunks = _split_text_chunks(payload.text, limit=500) if payload.chunk_text else [payload.text]

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
            profile_data = profile_meta or _load_voice_profile(payload.voice_profile)
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
            ref_audio_input = _prepare_ref_audio(payload.ref_audio)
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

    media_type = "audio/wav"
    filename = f"qwen3_tts_{payload.mode}.wav"

    return StreamingResponse(
        buffer,
        media_type=media_type,
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "model": DEFAULT_MODEL_ID, "device": DEFAULT_DEVICE}


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
