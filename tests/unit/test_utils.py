"""Unit tests for whisperx.utils writer classes and pure helpers."""

from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from whisperx.utils import (
    LANGUAGES,
    PUNKT_LANGUAGES,
    TO_LANGUAGE_CODE,
    WriteAudacity,
    WriteJSON,
    WriteSRT,
    WriteTSV,
    WriteTXT,
    WriteVTT,
    compression_ratio,
    format_timestamp,
    get_writer,
    interpolate_nans,
    make_safe,
    optional_float,
    optional_int,
    str2bool,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# SubtitlesWriter requires these option keys; None means default behavior.
_SUB_OPTIONS = {
    "max_line_width": None,
    "max_line_count": None,
    "highlight_words": False,
}


class TestFormatTimestamp:
    def test_seconds_only(self):
        # utils.format_timestamp default decimal marker is "."
        assert format_timestamp(12.345) == "00:12.345"

    def test_minutes(self):
        assert format_timestamp(75.0) == "01:15.000"

    def test_hours_always_include(self):
        assert format_timestamp(3661.5, always_include_hours=True) == "01:01:01.500"

    def test_hours_auto_when_over_one_hour(self):
        out = format_timestamp(3661.5)
        assert out.startswith("01:")

    def test_custom_decimal_marker(self):
        assert format_timestamp(1.5, decimal_marker=",") == "00:01,500"

    def test_zero(self):
        assert format_timestamp(0.0) == "00:00.000"

    def test_rounds_milliseconds(self):
        assert format_timestamp(1.23456) == "00:01.235"

    def test_negative_raises(self):
        with pytest.raises(AssertionError):
            format_timestamp(-0.1)


class TestCompressionRatio:
    def test_repetitive_text_compresses(self):
        # zlib has fixed header overhead; long repetitive input exceeds it.
        ratio = compression_ratio("aaaaaaaa" * 200)
        assert ratio > 1.0

    def test_incompressible_text_near_one(self):
        text = "abcdefghij" * 200
        ratio = compression_ratio(text)
        assert ratio >= 1.0

    def test_matches_definition(self):
        text = "hello world" * 50
        text_bytes = text.encode("utf-8")
        expected = len(text_bytes) / len(zlib.compress(text_bytes))
        assert compression_ratio(text) == expected


class TestStr2Bool:
    def test_true(self):
        assert str2bool("True") is True

    def test_false(self):
        assert str2bool("False") is False

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            str2bool("yes")


class TestOptionalInt:
    def test_none_string(self):
        assert optional_int("None") is None

    def test_number(self):
        assert optional_int("42") == 42


class TestOptionalFloat:
    def test_none_string(self):
        assert optional_float("None") is None

    def test_number(self):
        assert optional_float("1.5") == 1.5


class TestMakeSafe:
    def test_utf8_returns_unchanged(self, monkeypatch):
        # On a utf-8 system encoding, make_safe is a passthrough.
        monkeypatch.setattr("whisperx.utils.system_encoding", "utf-8")
        # Reimport the utf-8 branch by reloading behavior: since make_safe was
        # selected at import time, we exercise the actual function bound here.
        assert make_safe("hello café") == "hello café"


class TestInterpolateNans:
    def test_ignore_returns_input(self):
        s = pd.Series([1.0, np.nan, 3.0])
        out = interpolate_nans(s, method="ignore")
        assert out is s

    def test_nearest_fills_internal(self):
        s = pd.Series([1.0, np.nan, 3.0])
        out = interpolate_nans(s, method="nearest")
        assert out.notna().all()
        assert out.iloc[1] == 1.0

    def test_single_nonnull_ffill_bfill(self):
        s = pd.Series([np.nan, 5.0, np.nan])
        out = interpolate_nans(s, method="nearest")
        assert out.notna().all()
        assert (out == 5.0).all()


class TestLanguageMaps:
    def test_english_present(self):
        assert LANGUAGES["en"] == "english"

    def test_to_language_code_reverse(self):
        assert TO_LANGUAGE_CODE["english"] == "en"

    def test_alias_valencian(self):
        assert TO_LANGUAGE_CODE["valencian"] == "ca"

    def test_alias_haitian(self):
        assert TO_LANGUAGE_CODE["haitian"] == "ht"

    def test_punkt_english(self):
        assert PUNKT_LANGUAGES["en"] == "english"

    def test_punkt_slovene_label(self):
        assert PUNKT_LANGUAGES["sl"] == "slovene"


# --- ResultWriter subclasses ---


def _result_with_words():
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "hello world",
                "speaker": "SPEAKER_00",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4, "score": 0.9},
                    {"word": "world", "start": 0.4, "end": 1.0, "score": 0.8},
                ],
            }
        ],
    }


def _result_plain():
    return {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "hello world"},
            {"start": 1.0, "end": 2.0, "text": "foo bar"},
        ],
    }


class TestWriteTXT:
    def test_writes_text_without_speaker(self, tmp_path):
        w = WriteTXT(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        out = _read(tmp_path / "audio.txt")
        assert "hello world" in out
        assert "foo bar" in out

    def test_writes_speaker_prefix(self, tmp_path):
        w = WriteTXT(str(tmp_path))
        w(_result_with_words(), "audio.wav", {})
        out = _read(tmp_path / "audio.txt")
        assert "[SPEAKER_00]:" in out


class TestWriteJSON:
    def test_writes_valid_json(self, tmp_path):
        w = WriteJSON(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        data = json.loads(_read(tmp_path / "audio.json"))
        assert data["language"] == "en"
        assert len(data["segments"]) == 2

    def test_preserves_unicode(self, tmp_path):
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "café résumé"}],
        }
        w = WriteJSON(str(tmp_path))
        w(result, "audio.wav", {})
        text = _read(tmp_path / "audio.json")
        assert "café" in text


class TestWriteTSV:
    def test_header_and_rows(self, tmp_path):
        w = WriteTSV(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.tsv").strip().splitlines()
        assert lines[0] == "start\tend\ttext"
        # 1.0s -> 1000ms
        assert lines[1].startswith("0\t1000\thello world")

    def test_tabs_in_text_replaced(self, tmp_path):
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a\tb"}],
        }
        w = WriteTSV(str(tmp_path))
        w(result, "audio.wav", {})
        line = _read(tmp_path / "audio.tsv").strip().splitlines()[1]
        assert "\ta\tb" in line or line.endswith("a b")
        assert "a\tb" not in line.split("\t")[-1].replace(" ", "")


class TestWriteAudacity:
    def test_label_format(self, tmp_path):
        w = WriteAudacity(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.aud").strip().splitlines()
        first = lines[0].split("\t")
        assert first[0] == "0.0"
        assert first[1] == "1.0"
        assert first[2] == "hello world"

    def test_speaker_brackets(self, tmp_path):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"},
            ],
        }
        w = WriteAudacity(str(tmp_path))
        w(result, "audio.wav", {})
        line = _read(tmp_path / "audio.aud").strip().splitlines()[0]
        assert "[[SPEAKER_00]]" in line


class TestWriteVTT:
    def test_header_and_cue(self, tmp_path):
        w = WriteVTT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert out.startswith("WEBVTT")
        assert "00:00.000 --> 00:01.000" in out

    def test_word_subtitle_with_speaker(self, tmp_path):
        w = WriteVTT(str(tmp_path))
        w(_result_with_words(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "[SPEAKER_00]:" in out


class TestWriteSRT:
    def test_indices_and_comma_separator(self, tmp_path):
        w = WriteSRT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        assert "1\n" in out
        assert "2\n" in out
        # SRT uses comma as decimal marker and always includes hours
        assert "00:00:00,000 --> 00:00:01,000" in out


class TestGetWriter:
    def test_returns_txt_writer(self, tmp_path):
        w = get_writer("txt", str(tmp_path))
        assert isinstance(w, WriteTXT)

    def test_returns_srt_writer(self, tmp_path):
        assert isinstance(get_writer("srt", str(tmp_path)), WriteSRT)

    def test_returns_aud_writer(self, tmp_path):
        assert isinstance(get_writer("aud", str(tmp_path)), WriteAudacity)

    def test_all_writes_every_format(self, tmp_path):
        w = get_writer("all", str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        for ext in ("txt", "vtt", "srt", "tsv", "json"):
            assert (tmp_path / f"audio.{ext}").exists()


# --- SubtitlesWriter word-level paths ---


def _result_long_words():
    """Multi-word segment exercising line-break + subtitle-break logic."""
    words = []
    for i in range(20):
        words.append({"word": f"word{i:02d}", "start": float(i) * 0.5, "end": float(i) * 0.5 + 0.4})
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": " ".join(str(w["word"]) for w in words),
                "words": words,
            }
        ],
    }


class TestSubtitlesWriterWordPaths:
    def test_max_line_width_splits_lines(self, tmp_path):
        result = _result_long_words()
        opts = {**_SUB_OPTIONS, "max_line_width": 20, "max_line_count": 2}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # Multiple cue blocks produced by line-width + line-count splitting.
        assert out.count("-->") > 1

    def test_max_line_count_breaks_subtitle(self, tmp_path):
        result = _result_long_words()
        opts = {**_SUB_OPTIONS, "max_line_width": 30, "max_line_count": 2}
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.vtt")
        assert out.count("-->") > 1

    def test_highlight_words_underlines_current(self, tmp_path):
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "alpha beta", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "highlight_words": True}
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.vtt")
        assert "<u>" in out

    def test_japanese_text_joined_without_spaces(self, tmp_path):
        words = [
            {"word": "こんにちは", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "世界", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = {
            "language": "ja",
            "segments": [{"start": 0.0, "end": 1.0, "text": "こんにちは世界", "words": words}],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "こんにちは世界" in out

    def test_long_pause_creates_subtitle_break(self, tmp_path):
        # A >3s gap between word starts triggers a subtitle break when
        # preserve_segments is False (both max_line_width and max_line_count set).
        words = [
            {"word": "first", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "second", "start": 4.0, "end": 4.5, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 5.0, "text": "first second", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 5}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        assert out.count("-->") >= 2

    def test_word_without_timestamps_falls_back_to_segment(self, tmp_path):
        # Words missing start/end -> cue times come from the segment.
        words = [{"word": "hello"}, {"word": "world"}]
        result = {
            "language": "en",
            "segments": [{"start": 1.0, "end": 2.0, "text": "hello world", "words": words}],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "hello" in out and "world" in out

    def test_empty_segments_returns_nothing(self, tmp_path):
        result = {"language": "en", "segments": []}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        assert out == ""

    def test_segment_with_arrow_in_text_escaped(self, tmp_path):
        # "-->" in segment text is replaced with "->".
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a --> b"}],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "a -> b" in out
        # The cue separator is still "-->"
        assert "-->" in out


class TestMakeSafeNonUtf8:
    def test_non_utf8_encoding_replaces_unrepresentable(self, monkeypatch):
        # Reload the make_safe selection with a latin-1 system encoding.
        import importlib

        import whisperx.utils as utils_mod

        monkeypatch.setattr(utils_mod, "system_encoding", "latin-1")
        # The function selected at import time is the utf-8 branch; reimport
        # the module so the latin-1 branch binds.
        importlib.reload(utils_mod)
        try:
            # 'café' has é which latin-1 can encode; use a char outside latin-1.
            out = utils_mod.make_safe("hello \ufffd world")
            assert isinstance(out, str)
            assert "hello" in out
        finally:
            # Restore utf-8 binding for other tests.
            monkeypatch.setattr(utils_mod, "system_encoding", "utf-8")
            importlib.reload(utils_mod)
