"""Unit tests for whisperx.schema TypedDicts (runtime dict behavior)."""

from __future__ import annotations

from typing_extensions import NotRequired  # noqa: F401  # assert import works
from whisperx.schema import (
    AlignedTranscriptionResult,
    ProgressCallback,
    SegmentData,
    SingleAlignedSegment,
    SingleCharSegment,
    SingleSegment,
    SingleWordSegment,
    TranscriptionResult,
)


class TestSingleSegment:
    def test_minimal_segment_dict(self):
        seg: SingleSegment = {"start": 0.0, "end": 1.5, "text": "hello"}
        assert seg["text"] == "hello"
        assert seg["start"] == 0.0
        assert seg["end"] == 1.5

    def test_segment_optional_speaker(self):
        seg: SingleSegment = {"start": 0.0, "end": 1.0, "text": "hi", "speaker": "SPEAKER_00"}
        assert seg["speaker"] == "SPEAKER_00"

    def test_segment_optional_avg_logprob(self):
        seg: SingleSegment = {"start": 0.0, "end": 1.0, "text": "hi", "avg_logprob": -0.3}
        assert seg["avg_logprob"] == -0.3


class TestSingleWordSegment:
    def test_word_with_score(self):
        w: SingleWordSegment = {"word": "the", "start": 0.1, "end": 0.2, "score": 0.9}
        assert w["score"] == 0.9

    def test_word_with_speaker(self):
        w: SingleWordSegment = {
            "word": "the",
            "start": 0.1,
            "end": 0.2,
            "score": 0.9,
            "speaker": "SPEAKER_01",
        }
        assert w["speaker"] == "SPEAKER_01"


class TestSingleCharSegment:
    def test_char_segment(self):
        c: SingleCharSegment = {"char": "a", "start": 0.0, "end": 0.05, "score": 0.8}
        assert c["char"] == "a"


class TestSegmentData:
    def test_segment_data_fields(self):
        d: SegmentData = {
            "clean_char": ["a", "b"],
            "clean_cdx": [0, 1],
            "clean_wdx": [0],
            "sentence_spans": [(0, 2)],
        }
        assert d["clean_char"] == ["a", "b"]
        assert d["sentence_spans"] == [(0, 2)]


class TestAlignedTranscriptionResult:
    def test_minimal_aligned_result(self):
        res: AlignedTranscriptionResult = {"segments": [], "word_segments": []}
        assert res["segments"] == []
        assert res["word_segments"] == []

    def test_aligned_with_language(self):
        res: AlignedTranscriptionResult = {
            "segments": [],
            "word_segments": [],
            "language": "en",
        }
        assert res["language"] == "en"


class TestTranscriptionResult:
    def test_minimal(self):
        res: TranscriptionResult = {"segments": [], "language": "en"}
        assert res["language"] == "en"


class TestSingleAlignedSegment:
    def test_full_aligned_segment(self):
        seg: SingleAlignedSegment = {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "chars": None,
        }
        assert seg["chars"] is None

    def test_with_words(self):
        seg: SingleAlignedSegment = {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 0.5, "score": 1.0}],
            "chars": None,
        }
        assert seg["words"][0]["word"] == "hi"


def test_progress_callback_type_alias_accepts_none():
    cb: ProgressCallback = None
    assert cb is None
