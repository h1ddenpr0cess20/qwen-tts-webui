"""Text processing helpers."""

import re
import textwrap
from typing import List

from fastapi import HTTPException


def wrap_transcript(
    text: str,
    max_width: int,
    max_lines: int,
    break_long_words: bool = False,
) -> str:
    """Wrap transcript text for video overlays.

    Args:
        text: Input text to wrap.
        max_width: Maximum characters per line.
        max_lines: Maximum number of lines before truncation.
        break_long_words: Whether to split long words when wrapping.

    Returns:
        Wrapped text with newlines inserted, or an empty string if input is blank.
    """

    cleaned = re.sub(r"\\s+", " ", text or "").strip()
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
    return "\\n".join(lines)


def contains_cjk(text: str) -> bool:
    """Check whether text contains CJK/Hangul characters.

    Args:
        text: Input text to inspect.

    Returns:
        True when CJK or Hangul characters are detected.
    """

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


def split_text_chunks(text: str, limit: int = 500) -> List[str]:
    """Split text into chunks without breaking words.

    Args:
        text: Text to split into chunks.
        limit: Maximum characters per chunk.

    Returns:
        List of chunked strings.

    Raises:
        HTTPException: If a single token exceeds the limit.
    """

    if len(text) <= limit:
        return [text]
    tokens = re.findall(r"\\S+\\s*", text)
    chunks: List[str] = []
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
