"""Voice profile persistence and management helpers."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from fastapi import HTTPException

from app.config import DEFAULT_CLONE_MODEL, DEFAULT_DEVICE, VOICE_PROFILE_DIR
from app.deps import torch
from app.schemas import VoiceProfileCreate
from app.services.model_loader import load_model
from app.utils.audio import prepare_ref_audio


def sanitize_name(name: str) -> str:
    """Sanitize a voice profile name for safe filenames.

    Args:
        name: Raw profile name.

    Returns:
        Sanitized profile name.

    Raises:
        HTTPException: If the name is invalid after sanitization.
    """

    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_")
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid profile name.")
    return safe


def normalize_profile_name(name: str) -> str:
    """Normalize a profile name by stripping a .pt suffix.

    Args:
        name: Profile name or filename.

    Returns:
        Normalized profile name without extension.
    """

    base = name[:-3] if name.lower().endswith(".pt") else name
    return sanitize_name(base)


def profile_paths(name: str) -> Tuple[Path, Path]:
    """Return the profile and metadata paths for a given name.

    Args:
        name: Profile name.

    Returns:
        Tuple containing the .pt file path and metadata JSON path.
    """

    safe = normalize_profile_name(name)
    return VOICE_PROFILE_DIR / f"{safe}.pt", VOICE_PROFILE_DIR / f"{safe}.meta.json"


def save_voice_profile(
    name: str,
    prompt_items: object,
    model_id: str,
    original_name: Optional[str] = None,
) -> None:
    """Persist a voice profile to disk.

    Args:
        name: Sanitized profile name.
        prompt_items: Prompt items created by the model.
        model_id: Model identifier used to create the profile.
        original_name: Original name before sanitization.
    """

    safe = sanitize_name(name)
    pt_path, meta_path = profile_paths(safe)
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


def load_voice_profile(name: str) -> Dict[str, object]:
    """Load a voice profile from disk.

    Args:
        name: Profile name.

    Returns:
        Loaded voice profile data.

    Raises:
        HTTPException: If the profile is missing or corrupted.
    """

    pt_path, _ = profile_paths(name)
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


def delete_voice_profile(name: str) -> None:
    """Delete a voice profile and its metadata.

    Args:
        name: Profile name.

    Raises:
        HTTPException: If the profile does not exist.
    """

    pt_path, meta_path = profile_paths(name)
    removed = False
    for path in (pt_path, meta_path):
        if path.exists():
            path.unlink()
            removed = True
    if not removed:
        raise HTTPException(status_code=404, detail="Voice profile not found.")


def list_voice_profiles() -> List[Dict[str, object]]:
    """List available voice profiles.

    Returns:
        Sorted list of profile metadata entries.
    """

    items: List[Dict[str, object]] = []
    for file in VOICE_PROFILE_DIR.glob("*.pt"):
        try:
            meta_path = file.with_suffix(".meta.json")
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                items.append(
                    {
                        "name": meta.get("name") or file.stem,
                        "original_name": meta.get("original_name")
                        or meta.get("name")
                        or file.stem,
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


def import_voice_profile_file(filename: str, content: bytes) -> Dict[str, object]:
    """Import a saved voice profile from a .pt file.

    Args:
        filename: Uploaded filename (used for sanitization).
        content: Raw bytes of the .pt file.

    Returns:
        Metadata describing the imported profile.
    """

    if not filename:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if not filename.lower().endswith(".pt"):
        raise HTTPException(status_code=400, detail="Only .pt files are supported.")
    if not content:
        raise HTTPException(status_code=400, detail="Empty profile file.")

    safe_name = normalize_profile_name(filename)
    VOICE_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    pt_path, meta_path = profile_paths(safe_name)
    tmp_path = pt_path.with_suffix(".pt.tmp")
    tmp_path.write_bytes(content)

    try:
        data = torch.load(tmp_path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pragma: no cover - defensive guard
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail="Invalid voice profile file. Upload a .pt exported from this app.",
        ) from exc

    if "prompt_items" not in data:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Corrupted voice profile file.")

    tmp_path.replace(pt_path)
    meta = {
        "name": safe_name,
        "original_name": data.get("original_name") or data.get("name") or safe_name,
        "model_id": data.get("model_id"),
        "saved_at": data.get("saved_at") or time.time(),
    }
    meta_path.write_text(json.dumps(meta))
    return {"status": "imported", **meta}


def create_voice_profile(payload: VoiceProfileCreate, model_id: str, device: str) -> Dict[str, object]:
    """Create and persist a new voice profile.

    Args:
        payload: Voice profile creation request.
        model_id: Model identifier to use for cloning.
        device: Device map to load the model on.

    Returns:
        Metadata describing the created profile.

    Raises:
        HTTPException: If inputs are invalid or profile creation fails.
    """

    model = load_model(model_id, device)
    if not payload.x_vector_only_mode and not payload.ref_text:
        raise HTTPException(
            status_code=422,
            detail="Voice profile needs 'ref_text' unless x_vector_only_mode is true.",
        )
    ref_audio_input = prepare_ref_audio(payload.ref_audio)
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_input,
        ref_text=payload.ref_text or "",
        x_vector_only_mode=payload.x_vector_only_mode,
    )
    safe_name = sanitize_name(payload.name)
    save_voice_profile(safe_name, prompt_items, model_id, original_name=payload.name)
    return {
        "status": "ok",
        "name": safe_name,
        "original_name": payload.name,
        "model_id": model_id,
    }


def resolve_profile_model_id(payload: VoiceProfileCreate) -> str:
    """Resolve the model ID to use for a profile request.

    Args:
        payload: Voice profile request payload.

    Returns:
        Model identifier to use.
    """

    return payload.model_id or DEFAULT_CLONE_MODEL


def resolve_profile_device(payload: VoiceProfileCreate) -> str:
    """Resolve the device to use for a profile request.

    Args:
        payload: Voice profile request payload.

    Returns:
        Device mapping string.
    """

    return payload.device or DEFAULT_DEVICE
