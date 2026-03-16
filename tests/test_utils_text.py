"""Tests for text processing utilities (app/utils/text.py).

These are pure functions with no ML dependencies, so they are tested
directly without mocking.
"""

import pytest
from fastapi import HTTPException

from app.utils.text import (
    contains_cjk,
    fit_transcript_to_box,
    split_text_chunks,
    wrap_transcript,
)


class TestContainsCJK:
    """contains_cjk detection tests."""

    def test_pure_ascii_returns_false(self):
        assert contains_cjk("Hello world 123") is False

    def test_empty_string_returns_false(self):
        assert contains_cjk("") is False

    def test_japanese_hiragana_detected(self):
        assert contains_cjk("\u3053\u3093\u306b\u3061\u306f") is True  # konnichiwa

    def test_japanese_katakana_detected(self):
        assert contains_cjk("\u30ab\u30bf\u30ab\u30ca") is True  # katakana

    def test_chinese_ideographs_detected(self):
        assert contains_cjk("\u4f60\u597d") is True  # nihao

    def test_korean_hangul_detected(self):
        assert contains_cjk("\uc548\ub155") is True  # annyeong

    def test_fullwidth_forms_detected(self):
        assert contains_cjk("\uff21\uff22\uff23") is True  # fullwidth ABC

    def test_cjk_symbols_detected(self):
        assert contains_cjk("\u3001\u3002") is True  # ideographic comma/period

    def test_mixed_latin_and_cjk_detected(self):
        assert contains_cjk("Hello \u4e16\u754c") is True

    def test_cjk_extension_a_detected(self):
        assert contains_cjk("\u3400") is True

    def test_hangul_jamo_detected(self):
        # U+1100 = Hangul Jamo range
        assert contains_cjk("\u1100") is True

    def test_katakana_phonetic_extensions_detected(self):
        assert contains_cjk("\u31f0") is True


class TestWrapTranscript:
    """wrap_transcript line wrapping tests."""

    def test_empty_input_returns_empty(self):
        assert wrap_transcript("", max_width=40, max_lines=5) == ""

    def test_none_input_returns_empty(self):
        assert wrap_transcript(None, max_width=40, max_lines=5) == ""

    def test_whitespace_only_returns_empty(self):
        assert wrap_transcript("   \n\t  ", max_width=40, max_lines=5) == ""

    def test_short_text_not_wrapped(self):
        result = wrap_transcript("Hello world", max_width=40, max_lines=5)
        assert result == "Hello world"
        assert "\n" not in result

    def test_wrapping_respects_max_width(self):
        text = "The quick brown fox jumps over the lazy dog and keeps running"
        result = wrap_transcript(text, max_width=20, max_lines=10)
        for line in result.split("\n"):
            assert len(line) <= 20

    def test_truncation_at_max_lines_with_ellipsis(self):
        text = " ".join(["word"] * 100)
        result = wrap_transcript(text, max_width=10, max_lines=3)
        lines = result.split("\n")
        assert len(lines) == 3
        assert lines[-1].endswith("...")

    def test_long_word_forces_break_long_words(self):
        # A single word longer than max_width should trigger break_long_words
        text = "supercalifragilisticexpialidocious"
        result = wrap_transcript(text, max_width=10, max_lines=10)
        assert result  # should not be empty
        for line in result.split("\n"):
            assert len(line) <= 10

    def test_whitespace_normalization(self):
        text = "Hello    \n\t  world"
        result = wrap_transcript(text, max_width=40, max_lines=5)
        assert result == "Hello world"

    def test_explicit_break_long_words(self):
        text = "ABCDEFGHIJKLMNOP"
        result = wrap_transcript(text, max_width=5, max_lines=10, break_long_words=True)
        for line in result.split("\n"):
            assert len(line) <= 5


class TestSplitTextChunks:
    """split_text_chunks boundary and sentence-splitting tests."""

    def test_short_text_returned_as_single_chunk(self):
        result = split_text_chunks("Hello world.", limit=500)
        assert result == ["Hello world."]

    def test_text_exactly_at_limit(self):
        text = "a" * 500
        result = split_text_chunks(text, limit=500)
        assert result == [text]

    def test_splits_on_sentence_boundaries(self):
        s1 = "A" * 250 + "."
        s2 = "B" * 250 + "."
        text = f"{s1} {s2}"
        result = split_text_chunks(text, limit=300)
        assert len(result) == 2
        assert result[0] == s1
        assert result[1] == s2

    def test_sentence_exceeding_limit_raises(self):
        text = "A" * 501 + ". Short."
        with pytest.raises(HTTPException) as exc_info:
            split_text_chunks(text, limit=500)
        assert exc_info.value.status_code == 400
        assert "sentence" in exc_info.value.detail.lower() or "word" in exc_info.value.detail.lower()

    def test_word_exceeding_limit_raises(self):
        # No sentence boundaries, single word too long
        text = "A" * 501
        with pytest.raises(HTTPException) as exc_info:
            split_text_chunks(text, limit=500)
        assert exc_info.value.status_code == 400

    def test_multiple_sentences_grouped_within_limit(self):
        sentences = ["Short one.", "Another short.", "Third one."]
        text = " ".join(sentences)
        result = split_text_chunks(text, limit=500)
        # All sentences fit in one chunk
        assert len(result) == 1
        assert result[0] == text

    def test_word_based_fallback_when_no_sentence_boundary(self):
        # Long text with no sentence-ending punctuation
        words = ["word"] * 200
        text = " ".join(words)
        result = split_text_chunks(text, limit=50)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 50

    def test_small_limit(self):
        text = "Hello. World. Foo."
        result = split_text_chunks(text, limit=10)
        assert all(len(c) <= 10 for c in result)


class TestFitTranscriptToBox:
    """fit_transcript_to_box scaling and layout tests."""

    def test_empty_text_returns_empty_with_defaults(self):
        text, font_size, spacing = fit_transcript_to_box(
            "", safe_width_px=500, safe_height_px=300,
            base_font_size=40, glyph_ratio=0.5, max_lines_cap=10,
        )
        assert text == ""
        assert font_size == 40
        assert spacing == int(40 * 0.08)

    def test_none_text_returns_empty(self):
        text, _, _ = fit_transcript_to_box(
            None, safe_width_px=500, safe_height_px=300,
            base_font_size=40, glyph_ratio=0.5, max_lines_cap=10,
        )
        assert text == ""

    def test_short_text_fits_without_shrinking(self):
        text, font_size, _ = fit_transcript_to_box(
            "Hi", safe_width_px=1000, safe_height_px=500,
            base_font_size=40, glyph_ratio=0.5, max_lines_cap=10,
        )
        assert text == "Hi"
        assert font_size == 40

    def test_long_text_shrinks_font(self):
        long_text = " ".join(["word"] * 200)
        _, font_size, _ = fit_transcript_to_box(
            long_text, safe_width_px=200, safe_height_px=100,
            base_font_size=60, glyph_ratio=0.5, max_lines_cap=4,
            min_font_size=20,
        )
        assert font_size < 60

    def test_font_size_clamped_to_min(self):
        huge_text = " ".join(["word"] * 1000)
        _, font_size, _ = fit_transcript_to_box(
            huge_text, safe_width_px=100, safe_height_px=50,
            base_font_size=80, glyph_ratio=0.5, max_lines_cap=5,
            min_font_size=30,
        )
        assert font_size >= 30

    def test_font_size_clamped_to_max(self):
        _, font_size, _ = fit_transcript_to_box(
            "Hi", safe_width_px=5000, safe_height_px=5000,
            base_font_size=200, glyph_ratio=0.5, max_lines_cap=20,
            max_font_size=96,
        )
        assert font_size <= 96

    def test_returns_three_element_tuple(self):
        result = fit_transcript_to_box(
            "Test text", safe_width_px=500, safe_height_px=300,
            base_font_size=40, glyph_ratio=0.5, max_lines_cap=10,
        )
        assert len(result) == 3
        text, font_size, line_spacing = result
        assert isinstance(text, str)
        assert isinstance(font_size, int)
        assert isinstance(line_spacing, int)
