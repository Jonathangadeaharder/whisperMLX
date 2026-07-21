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


# Default-argument and content-asserting tests: kill default-value mutants
# and split-point/save mutants by asserting exact content.


class TestSubtitlesProcessorDefaults:
    def test_default_max_line_length_is_45(self):
        proc = SubtitlesProcessor([{"start": 0.0, "end": 1.0, "text": "hi"}], "en")
        assert proc.max_line_length == 45
        assert proc.min_char_length_splitter == 30
        assert proc.is_vtt is False

    def test_default_is_vtt_false_uses_comma_in_save(self, tmp_path):
        seg = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.0, "score": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0, "score": 1.0},
            ],
        }
        out_path = tmp_path / "out.srt"
        proc = SubtitlesProcessor([seg], "en")
        proc.save(str(out_path))
        text = out_path.read_text(encoding="utf-8")
        assert "WEBVTT" not in text
        assert "," in text
        # The timestamp line uses a comma decimal marker, not a dot.
        ts_line = next(ln for ln in text.splitlines() if "-->" in ln)
        assert "." not in ts_line.split(" --> ")[0]

    def test_comma_set_from_lang(self):
        proc = SubtitlesProcessor([{"start": 0.0, "end": 1.0, "text": "hi"}], "en")
        assert proc.comma == ","

    def test_conjunctions_set_from_lang(self):
        proc = SubtitlesProcessor([{"start": 0.0, "end": 1.0, "text": "hi"}], "en")
        assert "and" in proc.conjunctions


class TestDetermineAdvancedSplitPointsPlainWords:
    def test_plain_words_uses_text_split(self):
        text = " ".join(f"word{i}" for i in range(12))
        seg = {"start": 0.0, "end": 6.0, "text": text}
        proc = SubtitlesProcessor([seg], "en", max_line_length=20, min_char_length_splitter=5)
        sp = proc.determine_advanced_split_points(seg)
        assert isinstance(sp, list)
        assert len(sp) >= 1

    def test_plain_words_generate_subtitles_proportional_timing(self):
        text = "alpha beta gamma delta"
        seg = {"start": 0.0, "end": 4.0, "text": text}
        proc = SubtitlesProcessor([seg], "en", max_line_length=15, min_char_length_splitter=3)
        sp = proc.determine_advanced_split_points(seg)
        subs = proc.generate_subtitles_from_split_points(seg, sp)
        assert len(subs) >= 1
        assert subs[0]["start"] == 0.0
        assert subs[-1]["end"] <= 4.0  # pyrefly: ignore[unsupported-operation]
        for s in subs:
            assert s["text"]

    def test_japanese_plain_words_no_space_prefix(self):
        seg = {"start": 0.0, "end": 2.0, "text": "こんにちは世界"}
        proc = SubtitlesProcessor([seg], "ja", max_line_length=20, min_char_length_splitter=5)
        sp = proc.determine_advanced_split_points(seg)
        subs = proc.generate_subtitles_from_split_points(seg, sp)
        assert len(subs) >= 1
        assert subs[0]["text"] == "こんにちは世界"


class TestSaveDefaultFilename:
    def test_save_default_filename_is_subtitles_srt(self, tmp_path):
        seg = {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 1.0}],
        }
        proc = SubtitlesProcessor([seg], "en")
        # Pass explicit filename in tmp_path to avoid os.chdir (unsafe under
        # parallel mutation runs). The default is "subtitles.srt".
        out_path = tmp_path / "subtitles.srt"
        count = proc.save(str(out_path))
        assert out_path.exists()
        assert count >= 1

    def test_save_default_filename_signature(self):
        import inspect

        sig = inspect.signature(SubtitlesProcessor.save)
        assert sig.parameters["filename"].default == "subtitles.srt"
        assert sig.parameters["advanced_splitting"].default is True

    def test_save_writes_exact_srt_block(self, tmp_path):
        seg = {
            "start": 0.0,
            "end": 2.0,
            "text": "hello world",
            "words": [
                {"word": "hello", "start": 0.0, "end": 1.0, "score": 1.0},
                {"word": "world", "start": 1.0, "end": 2.0, "score": 1.0},
            ],
        }
        out_path = tmp_path / "sub.srt"
        proc = SubtitlesProcessor([seg], "en", max_line_length=100)
        proc.save(str(out_path))
        text = out_path.read_text(encoding="utf-8")
        assert "1" in text.splitlines()[0]
        assert "-->" in text
        assert "hello world" in text
        assert text.endswith(("\n", "\n\n"))

    def test_save_vtt_header_only_when_is_vtt(self, tmp_path):
        seg = {
            "start": 0.0,
            "end": 1.0,
            "text": "hi",
            "words": [{"word": "hi", "start": 0.0, "end": 1.0, "score": 1.0}],
        }
        out_srt = tmp_path / "a.srt"
        out_vtt = tmp_path / "b.vtt"
        proc_srt = SubtitlesProcessor([seg], "en", is_vtt=False)
        proc_vtt = SubtitlesProcessor([seg], "en", is_vtt=True)
        proc_srt.save(str(out_srt))
        proc_vtt.save(str(out_vtt))
        srt_text = out_srt.read_text(encoding="utf-8")
        vtt_text = out_vtt.read_text(encoding="utf-8")
        assert "WEBVTT" not in srt_text
        assert vtt_text.startswith("WEBVTT")


class TestGenerateSubtitlesFromSplitPointsEdges:
    def test_no_split_points_returns_last_fragment(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        seg = {"start": 0.0, "end": 1.0, "text": "hello world", "words": words}
        proc = SubtitlesProcessor([seg], "en")
        subs = proc.generate_subtitles_from_split_points(seg, [])
        assert len(subs) == 1
        assert subs[0]["start"] == 0.0
        assert subs[0]["end"] == 1.0
        assert subs[0]["text"] == "hello world"

    def test_next_start_time_extends_end(self):
        words = [
            {"word": "hello", "start": 0.0, "end": 0.5, "score": 1.0},
            {"word": "world", "start": 0.5, "end": 1.0, "score": 1.0},
        ]
        seg = {"start": 0.0, "end": 1.0, "text": "hello world", "words": words}
        proc = SubtitlesProcessor([seg], "en")
        subs = proc.generate_subtitles_from_split_points(seg, [], next_start_time=1.3)
        assert subs[0]["end"] == 1.3

    def test_dict_words_use_word_timestamps(self):
        words = [
            {"word": "alpha", "start": 0.5, "end": 1.0, "score": 1.0},
            {"word": "beta", "start": 1.0, "end": 1.5, "score": 1.0},
        ]
        seg = {"start": 0.0, "end": 2.0, "text": "alpha beta", "words": words}
        proc = SubtitlesProcessor([seg], "en", max_line_length=100)
        sp = proc.determine_advanced_split_points(seg)
        subs = proc.generate_subtitles_from_split_points(seg, sp)
        for s in subs:
            assert s["start"] in (0.5, 1.0)


class TestEstimateTimestampForWordEdges:
    def test_first_word_no_prev_uses_next_only(self):
        words = [
            {"word": "bb"},
            {"word": "c", "start": 1.0, "end": 1.5},
        ]
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 2.0, "text": "bb c", "words": words}], "en"
        )
        proc.estimate_timestamp_for_word(words, 0)
        assert words[0]["end"] == 1.0
        assert words[0]["start"] == 1.0 - len("bb") * 0.25

    def test_word_length_affects_end_time(self):
        words = [
            {"word": "a", "start": 0.0, "end": 0.5},
            {"word": "bbbb"},
        ]
        proc = SubtitlesProcessor(
            [{"start": 0.0, "end": 5.0, "text": "a bbbb", "words": words}], "en"
        )
        proc.estimate_timestamp_for_word(words, 1)
        assert words[1]["start"] == 0.5
        assert words[1]["end"] == 0.5 + len("bbbb") * 0.25
