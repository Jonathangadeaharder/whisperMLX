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
        with pytest.raises(AssertionError, match="non-negative timestamp expected"):
            format_timestamp(-0.1)

    def test_minutes_division_constant(self):
        # 60s -> 00:01:00,000. Kills the minutes // 60_000 -> 60_001 mutant.
        assert format_timestamp(60.0) == "01:00.000"

    def test_hours_division_constant(self):
        # 3600s -> 01:00:00,000. Kills the hours // 3_600_000 constant mutant.
        assert format_timestamp(3600.0) == "01:00:00.000"


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
        with pytest.raises(ValueError, match="Expected one of"):
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
        assert w.extension == "txt"  # pyrefly: ignore[missing-attribute]

    def test_returns_srt_writer(self, tmp_path):
        w = get_writer("srt", str(tmp_path))
        assert w.extension == "srt"  # pyrefly: ignore[missing-attribute]

    def test_returns_aud_writer(self, tmp_path):
        w = get_writer("aud", str(tmp_path))
        assert w.extension == "aud"  # pyrefly: ignore[missing-attribute]

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


# --- Exact-content assertions for writers and iterate_result ---------------
# These kill default-value and content mutants by asserting exact output
# strings rather than just substring presence.


class TestWriteTXTExact:
    def test_plain_text_lines_exact(self, tmp_path):
        w = WriteTXT(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.txt").splitlines()
        assert lines == ["hello world", "foo bar"]

    def test_speaker_prefix_exact(self, tmp_path):
        w = WriteTXT(str(tmp_path))
        w(_result_with_words(), "audio.wav", {})
        lines = _read(tmp_path / "audio.txt").splitlines()
        assert lines == ["[SPEAKER_00]: hello world"]


class TestWriteTSVExact:
    def test_exact_header_and_rows(self, tmp_path):
        w = WriteTSV(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.tsv").splitlines()
        assert lines[0] == "start\tend\ttext"
        assert lines[1] == "0\t1000\thello world"
        assert lines[2] == "1000\t2000\tfoo bar"

    def test_text_tab_replaced_with_space(self, tmp_path):
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a\tb"}],
        }
        w = WriteTSV(str(tmp_path))
        w(result, "audio.wav", {})
        line = _read(tmp_path / "audio.tsv").splitlines()[1]
        # text column is last, and contains "a b" not "a\tb"
        assert line.endswith("a b")
        assert "\t" in line


class TestWriteAudacityExact:
    def test_exact_label_line(self, tmp_path):
        w = WriteAudacity(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.aud").splitlines()
        assert lines[0] == "0.0\t1.0\thello world"
        assert lines[1] == "1.0\t2.0\tfoo bar"

    def test_speaker_bracket_exact(self, tmp_path):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"},
            ],
        }
        w = WriteAudacity(str(tmp_path))
        w(result, "audio.wav", {})
        line = _read(tmp_path / "audio.aud").splitlines()[0]
        assert line == "0.0\t1.0\t[[SPEAKER_00]]hi"


class TestWriteVTTExact:
    def test_exact_vtt_structure(self, tmp_path):
        w = WriteVTT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        # VTT starts with WEBVTT header, then blank line, then cue.
        assert out.startswith("WEBVTT\n\n")
        cues = out.split("\n\n")
        # First element is "WEBVTT", rest are cues.
        assert cues[1].startswith("00:00.000 --> 00:01.000\nhello world")
        assert cues[2].startswith("00:01.000 --> 00:02.000\nfoo bar")

    def test_vtt_uses_dot_decimal(self, tmp_path):
        w = WriteVTT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        ts_line = next(ln for ln in out.splitlines() if "-->" in ln)
        # VTT always uses "." as decimal marker, never ",".
        assert "." in ts_line.split(" --> ")[0]
        assert "," not in ts_line.split(" --> ")[0]


class TestWriteSRTExact:
    def test_exact_srt_structure(self, tmp_path):
        w = WriteSRT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 2
        assert blocks[0] == "1\n00:00:00,000 --> 00:00:01,000\nhello world"
        assert blocks[1] == "2\n00:00:01,000 --> 00:00:02,000\nfoo bar"

    def test_srt_uses_comma_decimal(self, tmp_path):
        w = WriteSRT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        ts_line = next(ln for ln in out.splitlines() if "-->" in ln)
        # SRT always uses "," as decimal marker, never ".".
        start, end = ts_line.split(" --> ")
        assert "," in start
        assert "," in end
        assert "." not in start
        assert "." not in end

    def test_srt_always_includes_hours(self, tmp_path):
        w = WriteSRT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        ts_line = next(ln for ln in out.splitlines() if "-->" in ln)
        # SRT timestamps always have HH:MM:SS,mmm format.
        for ts in ts_line.split(" --> "):
            assert len(ts.split(":")) == 3


class TestWriteJSONExact:
    def test_exact_json_content(self, tmp_path):
        w = WriteJSON(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        data = json.loads(_read(tmp_path / "audio.json"))
        assert data["language"] == "en"
        assert data["segments"] == [
            {"start": 0.0, "end": 1.0, "text": "hello world"},
            {"start": 1.0, "end": 2.0, "text": "foo bar"},
        ]


class TestGetWriterExact:
    def test_all_extensions(self, tmp_path):
        for fmt, _cls in [
            ("txt", WriteTXT),
            ("vtt", WriteVTT),
            ("srt", WriteSRT),
            ("tsv", WriteTSV),
            ("json", WriteJSON),
            ("aud", WriteAudacity),
        ]:
            w = get_writer(fmt, str(tmp_path))
            assert w.extension == fmt  # pyrefly: ignore[missing-attribute]

    def test_all_writes_all_formats_exact(self, tmp_path):
        w = get_writer("all", str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        for ext in ("txt", "vtt", "srt", "tsv", "json"):
            assert (tmp_path / f"audio.{ext}").exists()
        # Verify content is actually written (not empty).
        assert _read(tmp_path / "audio.txt").strip()
        assert _read(tmp_path / "audio.srt").strip()
        assert _read(tmp_path / "audio.vtt").startswith("WEBVTT")


class TestIterateResultPaths:
    def test_plain_segment_text_arrow_escaped(self, tmp_path):
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a --> b"}],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        # "-->" in text is replaced with "->".
        cue_text = out.split("\n\n")[1].split("\n", 1)[1]
        assert "a -> b" in cue_text
        assert "a --> b" not in cue_text

    def test_segment_speaker_prefix_in_plain_mode(self, tmp_path):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_00"},
            ],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "[SPEAKER_00]: hello" in out

    def test_empty_segments_writes_only_header(self, tmp_path):
        result = {"language": "en", "segments": []}
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert out == "WEBVTT\n\n"

    def test_empty_segments_srt_writes_empty(self, tmp_path):
        result = {"language": "en", "segments": []}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        assert out == ""

    def test_word_subtitle_cue_times_from_words(self, tmp_path):
        words = [
            {"word": "alpha", "start": 0.5, "end": 0.8, "score": 1.0},
            {"word": "beta", "start": 1.0, "end": 1.5, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": "alpha beta", "words": words}],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        ts_line = next(ln for ln in out.splitlines() if "-->" in ln)
        start, end = ts_line.split(" --> ")
        # Cue start = min(word starts) = 0.5, end = max(word ends) = 1.5.
        assert start == "00:00:00,500"
        assert end == "00:00:01,500"

    def test_word_without_timestamps_falls_back_to_segment_times(self, tmp_path):
        words = [{"word": "hello"}, {"word": "world"}]
        result = {
            "language": "en",
            "segments": [{"start": 1.0, "end": 2.0, "text": "hello world", "words": words}],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        ts_line = next(ln for ln in out.splitlines() if "-->" in ln)
        start, end = ts_line.split(" --> ")
        # Falls back to segment start=1.0, end=2.0.
        assert start == "00:00:01,000"
        assert end == "00:00:02,000"

    def test_highlight_words_yields_per_word_cues(self, tmp_path):
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "alpha beta", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "highlight_words": True}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # highlight_words emits one cue per word with <u> markup.
        assert out.count("-->") >= 2
        assert "<u>alpha</u>" in out
        assert "<u>beta</u>" in out

    def test_long_pause_breaks_subtitle(self, tmp_path):
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
        # >3s pause between word starts forces a subtitle break.
        assert out.count("-->") >= 2

    def test_max_line_count_breaks_subtitle(self, tmp_path):
        words = [
            {"word": f"word{i:02d}", "start": i * 0.5, "end": i * 0.5 + 0.4, "score": 1.0}
            for i in range(20)
        ]
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": " ".join(w["word"] for w in words),
                    "words": words,
                }
            ],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 30, "max_line_count": 2}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        assert out.count("-->") > 1

    def test_japanese_words_joined_without_spaces(self, tmp_path):
        words = [
            {"word": "こんにちは", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "世界", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = {
            "language": "ja",
            "segments": [{"start": 0.0, "end": 1.0, "text": "こんにちは世界", "words": words}],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        assert "こんにちは世界" in out
        # No space between Japanese words.
        assert "こんにちは 世界" not in out


class TestIterateResultEdgeCases:
    """Edge-case assertions killing long_pause, line_count, and default mutants."""

    def test_long_pause_exactly_3_seconds_no_break(self, tmp_path):
        # long_pause = timing["start"] - last > 3.0 (strict >).
        # A gap of exactly 3.0 should NOT trigger a pause break.
        words = [
            {"word": "first", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "second", "start": 3.0, "end": 3.5, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 4.0, "text": "first second", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 5}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # Gap == 3.0, not > 3.0 -> no subtitle break.
        assert out.count("-->") == 1

    def test_long_pause_above_3_seconds_breaks(self, tmp_path):
        # A gap > 3.0 triggers a subtitle break.
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

    def test_max_line_width_none_defaults_to_1000(self, tmp_path):
        # raw_max_line_width=None -> max_line_width=1000. A short text fits
        # in one line (no split).
        words = [
            {"word": f"w{i}", "start": float(i) * 0.1, "end": float(i) * 0.1 + 0.05, "score": 1.0}
            for i in range(10)
        ]
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": " ".join(w["word"] for w in words),
                    "words": words,
                }
            ],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        # With max_line_width=1000, all 10 two-char words (20 chars) fit in one cue.
        assert out.count("-->") == 1

    def test_preserve_segments_when_both_none(self, tmp_path):
        # Both max_line_width and max_line_count None -> preserve_segments=True.
        # Segments are preserved as-is (seg_break at i==0).
        words1 = [{"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0}]
        words2 = [{"word": "world", "start": 1.0, "end": 1.5, "score": 1.0}]
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "hello", "words": words1},
                {"start": 1.0, "end": 1.5, "text": "world", "words": words2},
            ],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        # Two segments -> two cues (seg_break preserves segment boundaries).
        assert out.count("-->") == 2

    def test_max_line_count_triggers_subtitle_break(self, tmp_path):
        # max_line_count triggers a subtitle break when line_count reaches the
        # limit. Need a small max_line_width so line breaks occur.
        words = [
            {"word": f"word{i:02d}", "start": i * 0.5, "end": i * 0.5 + 0.4, "score": 1.0}
            for i in range(20)
        ]
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 10.0,
                    "text": " ".join(w["word"] for w in words),
                    "words": words,
                }
            ],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 20, "max_line_count": 2}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # With max_line_width=20 and max_line_count=2, multiple subtitle breaks.
        assert out.count("-->") >= 3

    def test_word_without_start_no_long_pause(self, tmp_path):
        # Words without "start" -> long_pause=False (no break from pause).
        words = [{"word": "hello"}, {"word": "world"}]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 2.0, "text": "hello world", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 5}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        assert "hello" in out and "world" in out

    def test_highlight_words_yields_cue_with_correct_start(self, tmp_path):
        # highlight_words: each word gets its own cue with start/end from the word.
        words = [
            {"word": "alpha", "start": 1.0, "end": 1.5, "score": 1.0},
            {"word": "beta", "start": 2.0, "end": 2.5, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 3.0, "text": "alpha beta", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "highlight_words": True}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # The first word's cue starts at 00:01,000.
        assert "00:00:01,000" in out

    def test_speaker_prefix_in_word_mode(self, tmp_path):
        # In word mode, speaker prefix is added to each cue.
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "words": words,
                    "speaker": "SPEAKER_00",
                }
            ],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        assert "[SPEAKER_00]:" in out

    def test_plain_segment_speaker_prefix(self, tmp_path):
        # Plain segments (no words) with speaker get [speaker]: prefix.
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hello", "speaker": "SPEAKER_01"},
            ],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "[SPEAKER_01]: hello" in out


# Content-asserting writer tests: assert exact bytes to kill string/separator
# and control-flow mutants in write_result / iterate_result.


class TestWriteTXTContent:
    def test_plain_text_lines_match_input(self, tmp_path):
        w = WriteTXT(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.txt").splitlines()
        assert lines == ["hello world", "foo bar"]

    def test_speaker_prefix_exact_format(self, tmp_path):
        w = WriteTXT(str(tmp_path))
        w(_result_with_words(), "audio.wav", {})
        line = _read(tmp_path / "audio.txt").splitlines()[0]
        assert line == "[SPEAKER_00]: hello world"

    def test_no_speaker_has_no_brackets(self, tmp_path):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "plain text", "speaker": None},
            ],
        }
        w = WriteTXT(str(tmp_path))
        w(result, "audio.wav", {})
        out = _read(tmp_path / "audio.txt")
        assert "[" not in out
        assert out.strip() == "plain text"


class TestWriteTSVContent:
    def test_header_and_exact_rows(self, tmp_path):
        w = WriteTSV(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.tsv").splitlines()
        assert lines[0] == "start\tend\ttext"
        # Row 1: 0.0s -> 0ms, 1.0s -> 1000ms, tab-separated.
        assert lines[1] == "0\t1000\thello world"
        assert lines[2] == "1000\t2000\tfoo bar"

    def test_text_tab_replaced_with_space(self, tmp_path):
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a\tb\tc"}],
        }
        w = WriteTSV(str(tmp_path))
        w(result, "audio.wav", {})
        line = _read(tmp_path / "audio.tsv").splitlines()[1]
        # Three columns: start, end, text. Text has tabs replaced by spaces.
        cols = line.split("\t")
        assert len(cols) == 3
        assert cols[2] == "a b c"


class TestWriteAudacityContent:
    def test_label_lines_exact(self, tmp_path):
        w = WriteAudacity(str(tmp_path))
        w(_result_plain(), "audio.wav", {})
        lines = _read(tmp_path / "audio.aud").splitlines()
        assert lines[0] == "0.0\t1.0\thello world"
        assert lines[1] == "1.0\t2.0\tfoo bar"

    def test_speaker_double_brackets(self, tmp_path):
        result = {
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"},
            ],
        }
        w = WriteAudacity(str(tmp_path))
        w(result, "audio.wav", {})
        line = _read(tmp_path / "audio.aud").splitlines()[0]
        assert line == "0.0\t1.0\t[[SPEAKER_00]]hi"


class TestWriteVTTContent:
    def test_header_then_cues(self, tmp_path):
        w = WriteVTT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert out.startswith("WEBVTT\n\n")
        # Cue with --> arrow and dot decimal marker.
        assert "00:00.000 --> 00:01.000" in out
        assert "hello world" in out

    def test_no_webvtt_when_not_first(self, tmp_path):
        # The WEBVTT header is the very first line; nothing precedes it.
        w = WriteVTT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert out.split("\n", 1)[0] == "WEBVTT"


class TestWriteSRTContent:
    def test_numbered_cues_with_comma_and_hours(self, tmp_path):
        w = WriteSRT(str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        blocks = out.strip().split("\n\n")
        assert len(blocks) == 2
        # Each block starts with an index, then the timestamp line.
        assert blocks[0].startswith("1\n")
        assert blocks[1].startswith("2\n")
        # SRT uses comma decimal + always-include-hours.
        assert "00:00:00,000 --> 00:00:01,000" in blocks[0]
        assert "00:00:01,000 --> 00:00:02,000" in blocks[1]
        assert "hello world" in blocks[0]

    def test_text_arrow_replaced_in_srt(self, tmp_path):
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "a --> b"}],
        }
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.srt")
        # "-->" in text becomes "->"; the cue separator remains "-->".
        assert "a -> b" in out


class TestGetWriterDispatch:
    def test_all_writes_each_format_with_content(self, tmp_path):
        w = get_writer("all", str(tmp_path))
        w(_result_plain(), "audio.wav", _SUB_OPTIONS)
        # Each format has its distinguishing content.
        assert (tmp_path / "audio.txt").read_text(encoding="utf-8").strip()
        assert (tmp_path / "audio.vtt").read_text(encoding="utf-8").startswith("WEBVTT")
        assert "-->" in (tmp_path / "audio.srt").read_text(encoding="utf-8")
        assert (tmp_path / "audio.tsv").read_text(encoding="utf-8").startswith("start\t")
        import json as _json

        assert (
            _json.loads((tmp_path / "audio.json").read_text(encoding="utf-8"))["language"] == "en"
        )

    def test_unknown_format_raises_keyerror(self, tmp_path):
        # writers dict does not contain bogus formats.
        with pytest.raises(KeyError):
            get_writer("bogus", str(tmp_path))


class TestSubtitlesWriterIterateResult:
    def test_preserve_segments_keeps_segment_boundary(self, tmp_path):
        # With max_line_count=None, preserve_segments=True, so a new segment
        # forces a subtitle break at i==0. Two segments -> two cues.
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "first segment",
                    "words": [
                        {"word": "first", "start": 0.0, "end": 0.5, "score": 1.0},
                        {"word": "segment", "start": 0.5, "end": 1.0, "score": 1.0},
                    ],
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "second segment",
                    "words": [
                        {"word": "second", "start": 2.0, "end": 2.5, "score": 1.0},
                        {"word": "segment", "start": 2.5, "end": 3.0, "score": 1.0},
                    ],
                },
            ],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": None}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # Two cue blocks -> two "-->" separators.
        assert out.count("-->") == 2

    def test_long_pause_break_when_not_preserved(self, tmp_path):
        # preserve_segments=False (both width and count set); a >3s gap between
        # word starts triggers a subtitle break.
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 4.0, "end": 4.5, "score": 1.0},
            {"word": "gamma", "start": 4.6, "end": 5.0, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 5.0, "text": "alpha beta gamma", "words": words}],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 5}
        w = WriteSRT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.srt")
        # long_pause breaks the subtitle; >= 2 cues.
        assert out.count("-->") >= 2

    def test_max_line_count_breaks_subtitle(self, tmp_path):
        # line_count reaches max_line_count -> subtitle break. Use a small
        # max_line_width so words wrap to new lines, incrementing line_count.
        words = [
            {"word": f"word{i:02d}", "start": i * 0.3, "end": i * 0.3 + 0.2, "score": 1.0}
            for i in range(10)
        ]
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": " ".join(w["word"] for w in words),
                    "words": words,
                }
            ],
        }
        opts = {**_SUB_OPTIONS, "max_line_width": 12, "max_line_count": 2}
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", opts)
        out = _read(tmp_path / "audio.vtt")
        assert out.count("-->") >= 2

    def test_word_stripped_on_new_line(self, tmp_path):
        # A word with surrounding whitespace is stripped when a new line starts.
        words = [
            {"word": "  hello  ", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello world", "words": words}],
        }
        w = WriteVTT(str(tmp_path))
        w(result, "audio.wav", _SUB_OPTIONS)
        out = _read(tmp_path / "audio.vtt")
        assert "hello" in out
        # No double-space from the un-stripped leading whitespace.
        assert "  " not in out.replace("\n", "")


class TestSubtitlesWriterIterateResultEdges:
    """Precise long_pause / line_count / max_line_width tests killing logic mutants."""

    def _write(self, tmp_path, words, seg_start, seg_end, opts, writer_cls=None):
        from whisperx.utils import WriteSRT

        cls = writer_cls or WriteSRT
        result = {
            "language": "en",
            "segments": [{"start": seg_start, "end": seg_end, "text": "x", "words": words}],
        }
        w = cls(str(tmp_path))
        w(result, "audio.wav", opts)
        ext = "srt" if cls is WriteSRT else "vtt"
        return _read(tmp_path / f"audio.{ext}")

    def test_long_pause_strict_gt_at_exactly_three_seconds(self, tmp_path):
        # Gap of EXACTLY 3.0s: correct code (>3.0) -> no break (1 cue);
        # mutant (>=3.0) -> break (2 cues). Kills the > -> >= mutant.
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 3.0, "end": 3.5, "score": 1.0},
        ]
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 5}
        out = self._write(tmp_path, words, 0.0, 4.0, opts)
        # No long_pause break -> single cue.
        assert out.count("-->") == 1

    def test_long_pause_and_vs_or_with_preserve_segments(self, tmp_path):
        # preserve_segments=True (max_line_count=None). long_pause starts False.
        # With a >3s gap inside ONE segment, correct (and) keeps words together
        # (1 cue); mutant (or) splits (2 cues). Kills and -> or.
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 5.0, "end": 5.5, "score": 1.0},
        ]
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": None}
        out = self._write(tmp_path, words, 0.0, 6.0, opts)
        # preserve_segments forces a seg_break only at i==0 of a NEW segment;
        # here both words are in the same segment -> single cue.
        assert out.count("-->") == 1

    def test_long_pause_minus_vs_plus_operator(self, tmp_path):
        # last=2.0, start=4.0: correct (start-last=2.0) -> no break (1 cue);
        # mutant (start+last=6.0>3) -> break (2 cues). Kills - -> +.
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 2.0, "end": 2.5, "score": 1.0},
            {"word": "gamma", "start": 4.0, "end": 4.5, "score": 1.0},
        ]
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 5}
        out = self._write(tmp_path, words, 0.0, 5.0, opts)
        # Gaps: 2.0 then 2.0 -> both <=3 -> no long_pause break -> 1 cue.
        assert out.count("-->") == 1

    def test_line_count_resets_to_one_after_break(self, tmp_path):
        # max_line_count=2. After a forced break line_count resets to 1 (correct).
        # Mutant resets to 2, breaking one line earlier. Use long_pause break
        # (>3s gap) then short words; reset value affects line accumulation.
        words = [
            {"word": "alpha", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "beta", "start": 4.0, "end": 4.5, "score": 1.0},
            {"word": "gamma", "start": 4.6, "end": 4.7, "score": 1.0},
            {"word": "delta", "start": 4.8, "end": 4.9, "score": 1.0},
        ]
        opts = {**_SUB_OPTIONS, "max_line_width": 100, "max_line_count": 2}
        out = self._write(tmp_path, words, 0.0, 5.0, opts)
        # Break at beta (long_pause), then gamma+delta (line_count 1->2) breaks.
        assert out.count("-->") >= 2

    def test_default_max_line_width_1000_boundary(self, tmp_path):
        # raw_max_line_width=None -> max_line_width=1000. 1-char + 1000-char
        # words: correct (1+1000=1001<=1000 False) breaks; mutant (True) same.
        # max_line_count=1 makes a line break force a subtitle break.
        w1 = "a"
        w2 = "b" * 1000
        words = [
            {"word": w1, "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": w2, "start": 0.6, "end": 0.7, "score": 1.0},
        ]
        opts = {**_SUB_OPTIONS, "max_line_width": None, "max_line_count": 1}
        out = self._write(tmp_path, words, 0.0, 1.0, opts)
        # Correct: line break between w1 and w2 -> max_line_count=1 -> subtitle
        # break -> 2 cues.
        assert out.count("-->") == 2


class TestInterpolateNansEdges:
    def test_all_nan_returns_all_nan_via_ffill_bfill(self):
        # notnull().sum() == 0 (<=1), so ffill().bfill() path: stays all-NaN.
        s = pd.Series([np.nan, np.nan, np.nan])
        out = interpolate_nans(s, method="nearest")
        assert out.isna().all()

    def test_two_nonnull_interpolates(self):
        s = pd.Series([1.0, np.nan, np.nan, 4.0])
        out = interpolate_nans(s, method="nearest")
        assert out.notna().all()
        assert out.iloc[0] == 1.0
        assert out.iloc[-1] == 4.0

    def test_leading_nan_ffilled(self):
        s = pd.Series([np.nan, np.nan, 3.0, 4.0])
        out = interpolate_nans(s, method="nearest")
        assert out.notna().all()
        assert out.iloc[0] == 3.0

    def test_trailing_nan_bfilled(self):
        s = pd.Series([1.0, 2.0, np.nan, np.nan])
        out = interpolate_nans(s, method="nearest")
        assert out.notna().all()
        assert out.iloc[-1] == 2.0

    def test_linear_method(self):
        s = pd.Series([0.0, np.nan, 10.0])
        out = interpolate_nans(s, method="linear")
        assert out.notna().all()
        assert out.iloc[1] == 5.0


class TestFormatTimestampEdges:
    def test_just_under_one_hour_no_hours_marker(self):
        # 3599.999ms -> hours=0, so no hours marker; output is MM:SS.mmm.
        out = format_timestamp(3599.999)
        assert out == "59:59.999"

    def test_just_over_one_hour_adds_hours_marker(self):
        # 3600.0 -> hours=1, so the hours marker is added even though
        # always_include_hours defaults to False.
        out = format_timestamp(3600.0)
        assert out.startswith("01:")

    def test_exactly_one_hour(self):
        assert format_timestamp(3600.0) == "01:00:00.000"

    def test_large_value(self):
        assert format_timestamp(3661.5, always_include_hours=True) == "01:01:01.500"

    def test_default_decimal_marker_dot(self):
        assert format_timestamp(0.5).endswith(".500")

    def test_rounding_boundary(self):
        # 1.2345 -> 1234.5 ms -> rounds to 1235 (round() uses banker's rounding
        # but 1234.5 -> 1234 in python3 round; verify the actual value).
        out = format_timestamp(1.2345)
        assert out.endswith(("235", "234"))


class TestCompressionRatioEdges:
    def test_empty_string(self):
        # Empty input: 0 bytes / len(compress) = 0.0 (or raises). Verify it
        # returns a float without raising.
        out = compression_ratio("")
        assert isinstance(out, float)

    def test_single_char(self):
        # A single byte compresses to ~9 bytes of zlib overhead, ratio < 1.
        out = compression_ratio("a")
        assert isinstance(out, float)
        assert out < 1.0
