"""Unit tests for whisperx.subtitles_processor.SubtitlesProcessor."""

from __future__ import annotations

import pytest
from whisperx.subtitles_processor import SubtitlesProcessor, format_timestamp, normal_round


class TestNormalRound:
    @pytest.mark.parametrize("n,expected", [(0.4, 0), (0.5, 1), (0.6, 1), (1.4, 1), (1.5, 2)])
    def test_rounds(self, n, expected):
        assert normal_round(n) == expected


class TestFormatTimestamp:
    def test_default_comma(self):
        assert format_timestamp(1.5) == "00:00:01,500"

    def test_vtt_dot(self):
        assert format_timestamp(1.5, is_vtt=True) == "00:00:01.500"

    def test_hours(self):
        assert format_timestamp(3661.5, is_vtt=True) == "01:01:01.500"

    def test_negative_raises(self):
        with pytest.raises(AssertionError):
            format_timestamp(-1.0)


def _word_segment(word, start=None, end=None):
    d = {"word": word}
    if start is not None:
        d["start"] = start
    if end is not None:
        d["end"] = end
    return d


class TestEstimateTimestampForWord:
    def test_uses_prev_end_and_next_start(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "b"},  # target
            {"word": "c", "start": 1.2, "end": 1.5},
        ]
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 2.0, "text": "a b c", "words": words}], "en"
        )
        proc.estimate_timestamp_for_word(words, 1)
        assert words[1]["start"] == 0.5
        assert words[1]["end"] == 1.2

    def test_prev_end_no_next_uses_next_segment_time(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "b"},
        ]
        proc = SubtitlesProcessor([{"start": 0.0, "end": 2.0, "text": "a b", "words": words}], "en")
        proc.estimate_timestamp_for_word(words, 1, next_segment_start_time=1.0)
        assert words[1]["start"] == 0.5
        # gap (1.0 - 0.5 = 0.5) <= 1 -> end = next_segment_start_time
        assert words[1]["end"] == 1.0

    def test_prev_end_no_next_large_gap(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "b"},
        ]
        proc = SubtitlesProcessor([{"start": 0.0, "end": 5.0, "text": "a b", "words": words}], "en")
        proc.estimate_timestamp_for_word(words, 1, next_segment_start_time=3.0)
        assert words[1]["start"] == 0.5
        # gap (3.0 - 0.5 = 2.5) > 1 -> end = next_segment_start_time - 0.5
        assert words[1]["end"] == 2.5

    def test_prev_end_no_next_no_segment_time(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "bbbbbb"},
        ]
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 5.0, "text": "a bbbbbbb", "words": words}], "en"
        )
        proc.estimate_timestamp_for_word(words, 1)
        assert words[1]["start"] == 0.5
        # end = start + len(word) * 0.25
        assert words[1]["end"] == 0.5 + len("bbbbbb") * 0.25

    def test_only_next_start_available(self):
        words = [
            {"word": "bb"},  # target, first
            {"word": "c", "start": 1.0, "end": 1.5},
        ]
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 2.0, "text": "bb c", "words": words}], "en"
        )
        proc.estimate_timestamp_for_word(words, 0)
        assert words[0]["end"] == 1.0
        assert words[0]["start"] == 1.0 - len("bb") * 0.25

    def test_next_segment_time_only(self):
        words = [{"word": "b"}]
        proc = SubtitlesProcessor([{"start": 0.0, "end": 2.0, "text": "b", "words": words}], "en")
        proc.estimate_timestamp_for_word(words, 0, next_segment_start_time=3.0)
        assert words[0]["start"] == 2.0
        assert words[0]["end"] == 2.5

    def test_no_neighbors_no_segment_time(self):
        words = [{"word": "b"}]
        proc = SubtitlesProcessor([{"start": 0.0, "end": 2.0, "text": "b", "words": words}], "en")
        proc.estimate_timestamp_for_word(words, 0)
        assert words[0]["start"] == 0
        assert words[0]["end"] == 0


class TestProcessSegments:
    def test_simple_passthrough_without_splitting(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        seg = {"start": 0.0, "end": 1.0, "text": "hello world", "words": words}
        proc = SubtitlesProcessor([seg], "en", max_line_length=200)
        out = proc.process_segments(advanced_splitting=False)
        assert len(out) == 1
        assert out[0]["text"] == "hello world"
        assert out[0]["start"] == 0.0
        assert out[0]["end"] == 1.0

    def test_fills_missing_word_timestamps_when_not_splitting(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "world"},  # missing start/end
            {"word": "foo", "start": 1.0, "end": 1.5, "score": 1.0},
        ]
        seg = {"start": 0.0, "end": 2.0, "text": "hello world foo", "words": words}
        proc = SubtitlesProcessor([seg], "en", max_line_length=200)
        proc.process_segments(advanced_splitting=False)
        assert "start" in words[1]
        assert "end" in words[1]

    def test_advanced_splitting_long_line(self):
        # A long single segment should be split at the midpoint.
        text = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        words = [
            {"word": w, "start": i * 0.5, "end": i * 0.5 + 0.5, "score": 1.0}
            for i, w in enumerate(text.split())
        ]
        seg = {"start": 0.0, "end": 6.0, "text": text, "words": words}
        proc = SubtitlesProcessor([seg], "en", max_line_length=20, min_char_length_splitter=5)
        out = proc.process_segments(advanced_splitting=True)
        assert len(out) >= 2
        # subtitles should not overlap backward in start times
        starts = [s["start"] for s in out]
        assert starts == sorted(starts)

    def test_split_at_comma(self):
        text = "one two three, four five six"
        words = [
            {"word": w, "start": i * 0.3, "end": i * 0.3 + 0.3, "score": 1.0}
            for i, w in enumerate(text.split())
        ]
        seg = {"start": 0.0, "end": 3.0, "text": text, "words": words}
        proc = SubtitlesProcessor([seg], "en", max_line_length=100, min_char_length_splitter=3)
        out = proc.process_segments(advanced_splitting=True)
        assert len(out) >= 2

    def test_split_at_conjunction(self):
        text = "one two three and four five six seven"
        words = [
            {"word": w, "start": i * 0.3, "end": i * 0.3 + 0.3, "score": 1.0}
            for i, w in enumerate(text.split())
        ]
        seg = {"start": 0.0, "end": 3.0, "text": text, "words": words}
        proc = SubtitlesProcessor([seg], "en", max_line_length=100, min_char_length_splitter=3)
        out = proc.process_segments(advanced_splitting=True)
        assert len(out) >= 2

    def test_plain_words_no_dict(self):
        # segments without 'words' fall back to text.split()
        seg = {"start": 0.0, "end": 2.0, "text": "one two three four five six"}
        proc = SubtitlesProcessor([seg], "en", max_line_length=20, min_char_length_splitter=3)
        out = proc.process_segments(advanced_splitting=True)
        assert len(out) >= 1
        for s in out:
            assert s["text"]
            assert s["end"] > s["start"]


class TestComplexScriptLanguage:
    def test_japanese_uses_shorter_lengths(self):
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 1.0, "text": "テスト"}], "ja", max_line_length=45
        )
        assert proc.max_line_length == 30
        assert proc.min_char_length_splitter == 20

    def test_chinese_uses_shorter_lengths(self):
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 1.0, "text": "测试"}], "zh", max_line_length=45
        )
        assert proc.max_line_length == 30


class TestSave:
    def test_writes_srt_file(self, tmp_path):
        seg = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world foo bar",
            "words": [
                {"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0},
                {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0},
                {"word": "foo", "start": 1.0, "end": 1.5, "score": 1.0},
                {"word": "bar", "start": 1.5, "end": 2.0, "score": 1.0},
            ],
        }
        out_path = tmp_path / "sub.srt"
        proc = SubtitlesProcessor([seg], "en", max_line_length=100)
        count = proc.save(str(out_path), advanced_splitting=True)
        text = out_path.read_text(encoding="utf-8")
        assert "WEBVTT" not in text  # SRT, not VTT
        assert "-->" in text
        assert count >= 1

    def test_writes_vtt_file(self, tmp_path):
        seg = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.0, "score": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0, "score": 1.0},
            ],
        }
        out_path = tmp_path / "sub.vtt"
        proc = SubtitlesProcessor([seg], "en", max_line_length=100, is_vtt=True)
        proc.save(str(out_path), advanced_splitting=True)
        text = out_path.read_text(encoding="utf-8")
        assert text.startswith("WEBVTT")
        assert "." in text  # VTT uses dot decimal

    def test_save_without_advanced_splitting(self, tmp_path):
        words = [
            {"word": "hello", "start": 0.0, "end": 1.0, "score": 1.0},
            {"word": "world", "start": 1.0, "end": 2.0, "score": 1.0},
        ]
        seg = {"start": 0.0, "end": 2.0, "text": "hello world", "words": words}
        out_path = tmp_path / "sub.srt"
        proc = SubtitlesProcessor([seg], "en")
        count = proc.save(str(out_path), advanced_splitting=False)
        # process_segments returns one subtitle in the non-advanced path.
        assert count == len(proc.process_segments(advanced_splitting=False))
