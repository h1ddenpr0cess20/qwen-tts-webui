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

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return ""
    tokens = re.findall(r"\S+", cleaned)
    if tokens and not break_long_words and max(len(tok) for tok in tokens) > max_width:
        break_long_words = True
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


def fit_transcript_to_box(
    text: str,
    safe_width_px: float,
    safe_height_px: float,
    base_font_size: int,
    glyph_ratio: float,
    max_lines_cap: int,
    line_spacing_ratio: float = 0.08,
    min_font_size: int = 20,
    max_font_size: int = 96,
    break_long_words: bool = False,
) -> tuple[str, int, int]:
    """Wrap and scale transcript text to fit within a rectangular box.

    The function estimates text width using a simple glyph width heuristic
    (glyph_ratio * font_size) and shrinks the font size iteratively until both
    width and height constraints are satisfied or a minimum font size is
    reached. It falls back to ellipsis truncation only after exhausting size
    reductions.

    Args:
        text: Input transcript.
        safe_width_px: Usable width in pixels.
        safe_height_px: Usable height in pixels.
        base_font_size: Starting font size before scaling.
        glyph_ratio: Estimated glyph width as a fraction of font size
            (e.g., ~0.5 for Latin, ~0.9 for CJK/fullwidth).
        max_lines_cap: Hard cap on number of lines.
        line_spacing_ratio: Spacing between lines as a fraction of font size.
        min_font_size: Lower bound for font size.
        max_font_size: Upper bound for font size.
        break_long_words: Whether to split long tokens when wrapping.

    Returns:
        (wrapped_text, chosen_font_size, line_spacing_px)
    """

    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return "", base_font_size, int(base_font_size * line_spacing_ratio)

    font_size = int(max(min_font_size, min(max_font_size, base_font_size)))

    def attempt(size: int) -> tuple[str, List[str], int, int]:
        line_spacing = max(0, int(size * line_spacing_ratio))
        line_height = size + line_spacing
        # dynamic lines based on available height
        max_lines_dynamic = max(2, min(max_lines_cap, int(safe_height_px // max(line_height, 1))))
        max_width_chars = max(6, int(safe_width_px // max(size * glyph_ratio, 1)))
        wrapped = wrap_transcript(
            cleaned,
            max_width=max_width_chars,
            max_lines=max_lines_dynamic,
            break_long_words=break_long_words,
        )
        lines = wrapped.splitlines() if wrapped else []
        return wrapped, lines, line_spacing, line_height

    for _ in range(10):
        wrapped, lines, line_spacing, line_height = attempt(font_size)
        if not lines:
            return "", font_size, line_spacing
        max_line_px = max((len(line) * font_size * glyph_ratio for line in lines), default=0)
        total_height_px = len(lines) * line_height
        fits_width = max_line_px <= safe_width_px + 1
        fits_height = total_height_px <= safe_height_px + 1
        if fits_width and fits_height:
            return wrapped, font_size, line_spacing

        # Compute shrink factor based on the tighter constraint
        width_ratio = safe_width_px / max(max_line_px, 1)
        height_ratio = safe_height_px / max(total_height_px, 1)
        factor = min(width_ratio, height_ratio, 0.95)
        new_size = int(max(min_font_size, font_size * factor))
        if new_size == font_size:  # cannot shrink further
            break
        font_size = new_size

    # Fallback with truncation at the final font size
    wrapped, _, line_spacing, _ = attempt(font_size)
    return wrapped, font_size, line_spacing


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
    """Split text into chunks, preferring sentence boundaries.

    Args:
        text: Text to split into chunks.
        limit: Maximum characters per chunk.

    Returns:
        List of chunked strings.

    Raises:
        HTTPException: If a single token or sentence exceeds the limit.
    """

    def _chunk_by_words(segment: str) -> List[str]:
        tokens = re.findall(r"\S+\s*", segment)
        if not tokens:
            return []
        parts: List[str] = []
        current_part = ""
        for tok in tokens:
            if len(tok) > limit:
                raise HTTPException(
                    status_code=400,
                    detail=f"A single word exceeds the {limit}-character limit: '{tok.strip()[:32]}'...",
                )
            if len(current_part) + len(tok) > limit:
                if current_part.strip():
                    parts.append(current_part.rstrip())
                current_part = tok
            else:
                current_part += tok
        if current_part.strip():
            parts.append(current_part.rstrip())
        return parts

    if len(text) <= limit:
        return [text]

    # Prefer splitting on sentence boundaries to avoid mid-sentence cuts.
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
    if len(sentences) <= 1:
        # No clear sentence boundaries; fall back to word-based chunking.
        return _chunk_by_words(text)

    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        if len(sentence) > limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"A single sentence exceeds the {limit}-character limit. "
                    "Shorten the sentence or disable chunking."
                ),
            )
        if not current:
            current = sentence
            continue
        if len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            chunks.append(current.strip())
            current = sentence

    if current.strip():
        chunks.append(current.strip())
    return chunks
