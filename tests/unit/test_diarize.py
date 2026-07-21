"""Unit tests for whisperx.diarize (IntervalTree, assign_word_speakers, Segment,
DiarizationPipeline clustering). MLX runtime + WeSpeaker/segmentation are
mocked; the pure clustering + interval-tree logic is the behavior under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from whisperx.diarize import (
    DiarizationPipeline,
    IntervalTree,
    Segment,
    assign_word_speakers,
)
from whisperx.schema import (
    AlignedTranscriptionResult,
    TranscriptionResult,
)


class TestSegment:
    def test_defaults(self):
        s = Segment(0.0, 1.0)
        assert s.start == 0.0
        assert s.end == 1.0
        assert s.speaker is None

    def test_with_speaker(self):
        s = Segment(0.0, 1.0, "SPEAKER_00")
        assert s.speaker == "SPEAKER_00"


class TestIntervalTree:
    def test_empty_tree(self):
        tree = IntervalTree([])
        assert tree.query(0.0, 1.0) == []
        assert tree.find_nearest(0.5) is None

    def test_single_interval_overlap(self):
        tree = IntervalTree([(0.0, 1.0, "SPEAKER_00")])
        result = tree.query(0.2, 0.5)
        assert len(result) == 1
        assert result[0][0] == "SPEAKER_00"
        assert result[0][1] == pytest.approx(0.3)

    def test_no_overlap(self):
        tree = IntervalTree([(0.0, 1.0, "SPEAKER_00")])
        assert tree.query(2.0, 3.0) == []

    def test_query_before_all_starts(self):
        tree = IntervalTree([(1.0, 2.0, "SPEAKER_00")])
        # end=0.5 < first start 1.0 -> right_idx=0 -> empty
        assert tree.query(0.0, 0.5) == []

    def test_multiple_speakers_intersection(self):
        tree = IntervalTree([(0.0, 2.0, "A"), (1.0, 3.0, "B")])
        result = tree.query(1.0, 2.0)
        speakers = {r[0] for r in result}
        assert speakers == {"A", "B"}

    def test_find_nearest(self):
        tree = IntervalTree([(0.0, 1.0, "A"), (5.0, 6.0, "B")])
        assert tree.find_nearest(0.5) == "A"
        assert tree.find_nearest(5.5) == "B"
        # 2.5 is closer to midpoint of (5,6)=5.5? |5.5-2.5|=3, |0.5-2.5|=2 -> A
        assert tree.find_nearest(2.5) == "A"

    def test_partial_overlap_only(self):
        # An interval that touches the query boundary but does not overlap
        # (ends exactly at query start) should not be returned.
        tree = IntervalTree([(0.0, 1.0, "A")])
        result = tree.query(1.0, 2.0)
        assert result == []

    def test_interval_starts_exactly_at_query_end_no_overlap(self):
        # overlaps = starts < end (mutant: <=). Interval [1,2], query(0,1):
        # correct: 1 < 1 False -> no overlap. mutant: 1 <= 1 True -> overlap.
        tree = IntervalTree([(1.0, 2.0, "SPEAKER_00")])
        result = tree.query(0.0, 1.0)
        assert result == []

    def test_interval_ends_exactly_at_query_start_no_overlap(self):
        # overlaps = ends > start (mutant: >=). Interval [0,1], query(1,2):
        # correct: 1 > 1 False -> no overlap. mutant: 1 >= 1 True -> overlap.
        tree = IntervalTree([(0.0, 1.0, "SPEAKER_00")])
        result = tree.query(1.0, 2.0)
        assert result == []

    def test_zero_intersection_not_returned(self):
        # intersection > 0 (mutant: >= 0). Zero-intersection only happens at
        # boundary touch, which overlaps already excludes. Verify full overlap.
        tree = IntervalTree([(0.0, 1.0, "SPEAKER_00")])
        result = tree.query(0.0, 1.0)
        # Full overlap: intersection = 1.0 > 0 -> returned.
        assert len(result) == 1

    def test_and_not_or_for_overlaps(self):
        # mutmut_21: & -> |. Need an interval where only one of the two
        # conditions is True so & excludes it but | includes it.
        # [0,0.4] query(0.5,1.5): starts=0<1.5 T, ends=0.4>0.5 F. & excludes.
        tree = IntervalTree([(0.0, 0.4, "EARLY"), (0.5, 1.0, "GOOD")])
        result = tree.query(0.5, 1.5)
        speakers = [r[0] for r in result]
        assert "EARLY" not in speakers
        assert "GOOD" in speakers

    def test_searchsorted_left_side(self):
        # right_idx = searchsorted(starts, end, side="left"). side="left" gives
        # right_idx=0 when end equals first start, triggering early empty return.
        tree = IntervalTree([(1.0, 2.0, "SPEAKER_00")])
        # end=1.0. searchsorted([1.0], 1.0, "left")=0 -> right_idx=0 -> empty.
        result = tree.query(0.0, 1.0)
        assert result == []

    def _diarize_df(self, rows):
        return pd.DataFrame(
            [
                {
                    "segment": Segment(s, e, spk),
                    "label": int(spk.split("_")[1]) if spk and "_" in spk else 0,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                }
                for s, e, spk in rows
            ]
        )

    def test_empty_transcript_returned_unchanged(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {"segments": [], "language": "en"}
        out = assign_word_speakers(df, result)
        assert out is result

    def test_none_df_returns_unchanged(self):
        # None df exercises the early-return guard; typed as the union param.
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(None, result)  # pyrefly: ignore[bad-argument-type]
        assert out is result

    def test_assigns_segment_speaker_by_overlap(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.1, "end": 0.9, "text": "hello"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_assigns_word_speakers(self):
        df = self._diarize_df([(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.4, "score": 1.0},
                        {"word": "world", "start": 0.6, "end": 1.0, "score": 1.0},
                    ],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result)
        # Return type is the union; runtime variant is the aligned form.
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        words = aligned["segments"][0]["words"]
        assert words[0]["speaker"] == "SPEAKER_00"
        assert words[1]["speaker"] == "SPEAKER_01"

    def test_fill_nearest_when_no_overlap(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 2.0, "end": 3.0, "text": "later"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_no_fill_nearest_leaves_speaker_unset(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 5.0, "end": 6.0, "text": "later"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=False)
        assert "speaker" not in out["segments"][0]

    def test_word_without_start_skipped(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        # Word deliberately omits start/end to test the skip path; would not
        # satisfy SingleWordSegment, so suppress the TypedDict check here.
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hi",
                    "words": [{"word": "hi"}],  # pyrefly: ignore[bad-typed-dict-key]
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result)
        # Return type is the union; runtime variant is the aligned form.
        aligned2: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        assert "speaker" not in aligned2["segments"][0]["words"][0]

    def test_speaker_embeddings_attached(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        emb = {"SPEAKER_00": [0.1, 0.2]}
        out = assign_word_speakers(df, result, speaker_embeddings=emb)
        assert out["speaker_embeddings"] == emb

    def test_dominant_speaker_chosen_on_overlap(self):
        # Two speakers overlap a segment; the one with the larger intersection wins.
        df = self._diarize_df([(0.0, 0.4, "SPEAKER_00"), (0.1, 1.0, "SPEAKER_01")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.1, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert out["segments"][0]["speaker"] == "SPEAKER_01"


class TestEstimateClusters:
    def test_returns_min_when_embeddings_too_few(self):
        # Single embedding -> pdist empty -> returns min_s.
        embs = np.array([[1.0, 0.0]], dtype=np.float32)
        n = DiarizationPipeline._estimate_clusters(embs, min_s=1, max_s=3)
        assert n == 1

    def test_picks_best_within_range(self):
        # Two well-separated clusters of identical embeddings.
        embs = np.array(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=np.float32,
        )
        n = DiarizationPipeline._estimate_clusters(embs, min_s=1, max_s=4)
        assert 1 <= n <= 4

    def test_max_s_bounds_result(self):
        embs = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        n = DiarizationPipeline._estimate_clusters(embs, min_s=1, max_s=2)
        assert n <= 2


class TestDiarizationPipelineCall:
    def _make_pipeline(self, monkeypatch, segment_result=None, embeddings=None):
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe._segment_audio = MagicMock(
            return_value=(
                np.array(segment_result if segment_result is not None else [[0.8], [0.9]]),
                np.array([0.5, 1.0]),
            )
        )
        # Return distinct embeddings per call so clustering has variety.
        emb_iter = iter(embeddings or [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

        def _embed(audio, weights=None):
            try:
                return np.array(next(emb_iter))
            except StopIteration:
                return np.array([1.0, 0.0])

        pipe._embed = MagicMock(side_effect=_embed)
        pipe._wespeaker_weights = {}
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363

        # Binarize returns multiple short speech segments so clustering has >=2 samples.
        monkeypatch.setattr(
            DiarizationPipeline,
            "_binarize_segments",
            lambda self, scores, frame_times: [(0.0, 1.0), (2.0, 3.0)],
        )
        return pipe

    def test_no_speech_returns_empty_df(self, monkeypatch):
        pipe = self._make_pipeline(monkeypatch)
        monkeypatch.setattr(DiarizationPipeline, "_binarize_segments", lambda self, s, f: [])
        out = pipe(np.zeros(16000, dtype=np.float32))
        assert isinstance(out, pd.DataFrame)
        assert len(out) == 0

    def test_clustering_assigns_speakers(self, monkeypatch):
        pipe = self._make_pipeline(monkeypatch)
        out = pipe(np.zeros(16000 * 4, dtype=np.float32), num_speakers=2)
        assert isinstance(out, pd.DataFrame)
        assert len(out) >= 2
        for spk in out["speaker"]:
            assert spk.startswith("SPEAKER_")

    def test_return_embeddings_dict(self, monkeypatch):
        pipe = self._make_pipeline(monkeypatch)
        out, embs = pipe(
            np.zeros(16000 * 4, dtype=np.float32), num_speakers=2, return_embeddings=True
        )
        assert isinstance(out, pd.DataFrame)
        assert isinstance(embs, dict)
        assert all(k.startswith("SPEAKER_") for k in embs)

    def test_progress_callback_called(self, monkeypatch):
        pipe = self._make_pipeline(monkeypatch)
        calls = []
        pipe(np.zeros(16000 * 4, dtype=np.float32), num_speakers=2, progress_callback=calls.append)
        # 50 (after segmentation), 80 (after embeddings), 100 (after clustering).
        assert 50.0 in calls
        assert 80.0 in calls
        assert 100.0 in calls

    def test_long_segment_split_into_windows(self, monkeypatch):
        # A 5s speech segment (> EMB_WINDOW=3) is split into 3s windows.
        pipe = self._make_pipeline(monkeypatch)
        monkeypatch.setattr(
            DiarizationPipeline, "_binarize_segments", lambda self, s, f: [(0.0, 5.0)]
        )
        out = pipe(np.zeros(16000 * 6, dtype=np.float32), num_speakers=2)
        # Multiple embeddings extracted -> multiple rows.
        assert len(out) >= 2

    def test_string_audio_loads_via_load_audio(self, monkeypatch, tmp_wav_factory):
        # 4 seconds of audio so both binarized segments (0-1s and 2-3s) fit.
        audio = np.zeros(16000 * 4, dtype=np.float32)
        path = tmp_wav_factory(audio)
        pipe = self._make_pipeline(monkeypatch)
        out = pipe(path, num_speakers=2)
        assert len(out) >= 2

    def test_embeddings_too_short_skipped(self, monkeypatch):
        # Segment shorter than 0.3s -> skipped -> no embeddings -> empty df.
        pipe = self._make_pipeline(monkeypatch)
        monkeypatch.setattr(
            DiarizationPipeline, "_binarize_segments", lambda self, s, f: [(0.0, 0.1)]
        )
        out = pipe(np.zeros(1600, dtype=np.float32), num_speakers=1)
        assert len(out) == 0


class TestBinarizeSegments:
    def test_returns_float_tuples(self, monkeypatch):
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363

        class FakeSeg:
            def __init__(self, start, end):
                self.start = start
                self.end = end

        with patch("whisperx.vads.pyannote._Binarize") as BinCls:
            inst = MagicMock()
            inst.return_value = [FakeSeg(0.0, 1.0), FakeSeg(2.0, 3.0)]
            BinCls.return_value = inst
            out = pipe._binarize_segments(np.array([0.9]), np.array([0.5]))
        assert out == [(0.0, 1.0), (2.0, 3.0)]
        assert all(isinstance(s, float) and isinstance(e, float) for s, e in out)


# Default-argument and real-data tests: kill default-value and construction
# mutants by exercising real interval data and real score arrays.


class TestAssignWordSpeakersDefaults:
    def _df(self, rows):
        return pd.DataFrame(
            [
                {
                    "segment": Segment(s, e, spk),
                    "label": 0,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                }
                for s, e, spk in rows
            ]
        )

    def test_default_fill_nearest_is_false(self):
        # Call with only required args; fill_nearest defaults to False, so a
        # segment with no overlap does NOT get a speaker assigned.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 5.0, "end": 6.0, "text": "later"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert "speaker" not in out["segments"][0]

    def test_default_returns_transcript_result(self):
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert out is result
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_empty_df_returns_unchanged(self):
        empty_df = pd.DataFrame(columns=["segment", "label", "speaker", "start", "end"])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(empty_df, result)
        assert "speaker" not in out["segments"][0]

    def test_fill_nearest_default_false_for_words_too(self):
        # Words with no overlap and fill_nearest=False stay unset.
        df = self._df([(0.0, 0.4, "SPEAKER_00")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hi there",
                    "words": [
                        {"word": "hi", "start": 0.0, "end": 0.3, "score": 1.0},
                        {"word": "there", "start": 0.9, "end": 1.0, "score": 1.0},
                    ],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        # "there" (0.9-1.0) does not overlap (0.0-0.4); fill_nearest=False default.
        assert "speaker" not in aligned["segments"][0]["words"][1]


class TestIntervalTreeRealData:
    def test_unsorted_intervals_are_sorted_by_start(self):
        # Pass intervals out of order; query still works because __init__ sorts.
        tree = IntervalTree([(5.0, 6.0, "B"), (0.0, 1.0, "A"), (2.0, 3.0, "C")])
        # starts array is sorted ascending.
        starts = list(tree.starts)
        assert starts == sorted(starts)
        assert starts == [0.0, 2.0, 5.0]
        # speakers follow the sorted order.
        assert tree.speakers == ["A", "C", "B"]

    def test_starts_and_ends_are_float64_arrays(self):
        tree = IntervalTree([(0.0, 1.0, "A")])
        assert tree.starts.dtype == np.float64
        assert tree.ends.dtype == np.float64
        assert tree.starts[0] == 0.0
        assert tree.ends[0] == 1.0

    def test_empty_tree_attributes(self):
        tree = IntervalTree([])
        assert len(tree.starts) == 0
        assert len(tree.ends) == 0
        assert tree.speakers == []
        assert tree.query(0.0, 1.0) == []
        assert tree.find_nearest(0.5) is None

    def test_query_returns_intersection_duration(self):
        tree = IntervalTree([(0.0, 4.0, "A")])
        # Query (1.0, 3.0) -> intersection = 2.0.
        result = tree.query(1.0, 3.0)
        assert len(result) == 1
        assert result[0][0] == "A"
        assert result[0][1] == pytest.approx(2.0)

    def test_query_partial_overlap_at_end(self):
        tree = IntervalTree([(0.0, 2.0, "A"), (3.0, 5.0, "B")])
        # Query (1.5, 3.5) overlaps both: A intersects [1.5,2.0)=0.5, B [3.0,3.5)=0.5.
        result = tree.query(1.5, 3.5)
        speakers = {r[0] for r in result}
        assert speakers == {"A", "B"}

    def test_find_nearest_picks_closest_midpoint(self):
        tree = IntervalTree([(0.0, 2.0, "A"), (10.0, 12.0, "B")])
        # midpoints: A=1.0, B=11.0. t=4.0 is closer to A (|1-4|=3 vs |11-4|=7).
        assert tree.find_nearest(4.0) == "A"
        assert tree.find_nearest(8.0) == "B"

    def test_query_end_exactly_at_start_returns_empty(self):
        tree = IntervalTree([(1.0, 2.0, "A")])
        # end=1.0 equals the interval start; no overlap (strict <).
        assert tree.query(0.0, 1.0) == []


class TestBinarizeSegmentsReal:
    def test_binarize_with_real_speech_scores(self):
        # Drive _binarize_segments with a real score array that crosses onset.
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363
        # Scores: 0.9 (speech), 0.9, 0.1 (below offset -> segment ends), 0.9 (speech).
        scores = np.array([0.9, 0.9, 0.1, 0.9], dtype=np.float32)
        frame_times = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float32)
        out = pipe._binarize_segments(scores, frame_times)
        # At least one speech segment produced.
        assert len(out) >= 1
        for s, e in out:
            assert isinstance(s, float)
            assert isinstance(e, float)
            assert e > s

    def test_binarize_uses_onset_and_offset_thresholds(self):
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363
        # All scores below onset -> no speech.
        scores = np.array([0.1, 0.2, 0.1], dtype=np.float32)
        frame_times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = pipe._binarize_segments(scores, frame_times)
        assert out == []

    def test_binarize_returns_float_tuples(self):
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363
        scores = np.array([[0.9], [0.9], [0.9]], dtype=np.float32)
        frame_times = np.array([0.0, 0.5, 1.0], dtype=np.float32)
        out = pipe._binarize_segments(scores, frame_times)
        assert len(out) >= 1
        for s, e in out:
            assert isinstance(s, float) and isinstance(e, float)


# --- Exact-value assertions for IntervalTree and assign_word_speakers -------
# These kill default-value, sorting, and intersection mutants by asserting
# exact speaker labels and intersection magnitudes.


class TestIntervalTreeExact:
    def test_intervals_sorted_by_start(self):
        tree = IntervalTree([(5.0, 6.0, "B"), (0.0, 1.0, "A"), (2.0, 3.0, "C")])
        # Internal sort must order by start; speakers list follows the sort.
        assert list(tree.speakers) == ["A", "C", "B"]
        assert tree.starts.tolist() == [0.0, 2.0, 5.0]
        assert tree.ends.tolist() == [1.0, 3.0, 6.0]

    def test_single_interval_exact_intersection(self):
        tree = IntervalTree([(0.0, 1.0, "SPEAKER_00")])
        result = tree.query(0.2, 0.5)
        assert len(result) == 1
        assert result[0] == ("SPEAKER_00", pytest.approx(0.3))

    def test_exact_intersection_is_min_end_minus_max_start(self):
        # intersection = min(end, q_end) - max(start, q_start)
        tree = IntervalTree([(0.0, 2.0, "A")])
        result = tree.query(0.5, 1.5)
        # intersection = min(2.0, 1.5) - max(0.0, 0.5) = 1.5 - 0.5 = 1.0
        assert result == [("A", pytest.approx(1.0))]

    def test_query_within_interval_returns_full_overlap(self):
        tree = IntervalTree([(0.0, 10.0, "A")])
        result = tree.query(2.0, 5.0)
        # intersection = min(10, 5) - max(0, 2) = 5 - 2 = 3
        assert result == [("A", pytest.approx(3.0))]

    def test_two_overlapping_speakers_exact(self):
        tree = IntervalTree([(0.0, 2.0, "A"), (1.0, 3.0, "B")])
        result = tree.query(1.0, 2.0)
        result_dict = {r[0]: r[1] for r in result}
        # A: min(2.0, 2.0) - max(0.0, 1.0) = 2.0 - 1.0 = 1.0
        assert result_dict["A"] == pytest.approx(1.0)
        # B: min(3.0, 2.0) - max(1.0, 1.0) = 2.0 - 1.0 = 1.0
        assert result_dict["B"] == pytest.approx(1.0)

    def test_boundary_touch_not_returned(self):
        # An interval ending exactly at query start has intersection 0 (excluded).
        tree = IntervalTree([(0.0, 1.0, "A")])
        result = tree.query(1.0, 2.0)
        assert result == []

    def test_find_nearest_returns_nearest_midpoint(self):
        tree = IntervalTree([(0.0, 2.0, "A"), (10.0, 12.0, "B")])
        # midpoints: A=1.0, B=11.0. time=3.0 -> |1-3|=2, |11-3|=8 -> A.
        assert tree.find_nearest(3.0) == "A"
        # time=8.0 -> |1-8|=7, |11-8|=3 -> B.
        assert tree.find_nearest(8.0) == "B"

    def test_find_nearest_exact_at_midpoint(self):
        tree = IntervalTree([(0.0, 2.0, "A")])
        # midpoint = 1.0; querying exactly at it returns A.
        assert tree.find_nearest(1.0) == "A"

    def test_query_end_before_all_starts_returns_empty(self):
        tree = IntervalTree([(5.0, 6.0, "A"), (10.0, 11.0, "B")])
        assert tree.query(0.0, 1.0) == []

    def test_empty_tree_find_nearest_returns_none(self):
        tree = IntervalTree([])
        assert tree.find_nearest(0.5) is None

    def test_three_intervals_speakers_preserved(self):
        tree = IntervalTree([(0.0, 1.0, "X"), (2.0, 3.0, "Y"), (4.0, 5.0, "Z")])
        assert tree.speakers == ["X", "Y", "Z"]
        result = tree.query(2.5, 4.5)
        speakers = {r[0] for r in result}
        assert speakers == {"Y", "Z"}


class TestAssignWordSpeakersExact:
    def _diarize_df(self, rows):
        return pd.DataFrame(
            [
                {
                    "segment": Segment(s, e, spk),
                    "label": int(spk.split("_")[1]) if spk and "_" in spk else 0,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                }
                for s, e, spk in rows
            ]
        )

    def test_default_fill_nearest_is_false(self):
        # fill_nearest defaults to False; segment with no overlap gets no speaker.
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 5.0, "end": 6.0, "text": "later"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert "speaker" not in out["segments"][0]

    def test_segment_speaker_exact_label(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.1, "end": 0.9, "text": "hello"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_word_speaker_exact_labels(self):
        df = self._diarize_df([(0.0, 0.5, "SPEAKER_00"), (0.5, 1.0, "SPEAKER_01")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.4, "score": 1.0},
                        {"word": "world", "start": 0.6, "end": 1.0, "score": 1.0},
                    ],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        words = aligned["segments"][0]["words"]
        assert words[0]["speaker"] == "SPEAKER_00"
        assert words[1]["speaker"] == "SPEAKER_01"

    def test_fill_nearest_assigns_nearest_speaker(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 2.0, "end": 3.0, "text": "later"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        # midpoint of segment (2,3) = 2.5; midpoint of interval (0,1) = 0.5.
        # Only one speaker, so nearest is SPEAKER_00.
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_fill_nearest_word_assigns_nearest(self):
        df = self._diarize_df([(0.0, 0.5, "SPEAKER_00")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 2.0,
                    "end": 3.0,
                    "text": "later",
                    "words": [
                        {"word": "later", "start": 2.0, "end": 3.0, "score": 1.0},
                    ],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        assert aligned["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"

    def test_dominant_speaker_is_max_intersection(self):
        # Two speakers overlap a segment; the one with the larger intersection wins.
        # A: (0.0, 0.4) -> intersection with (0.1, 1.0) = min(0.4,1.0)-max(0.0,0.1)=0.3
        # B: (0.1, 1.0) -> intersection with (0.1, 1.0) = min(1.0,1.0)-max(0.1,0.1)=0.9
        df = self._diarize_df([(0.0, 0.4, "SPEAKER_00"), (0.1, 1.0, "SPEAKER_01")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.1, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert out["segments"][0]["speaker"] == "SPEAKER_01"

    def test_word_without_end_uses_start_as_end(self):
        # word_end = word.get("end", word_start); a word with only start still
        # queries with start==end, which yields zero intersection, so no speaker.
        # Use fill_nearest to verify the fallback uses word_mid = start.
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hi",
                    "words": [{"word": "hi", "start": 0.2, "score": 1.0}],  # pyrefly: ignore[bad-typed-dict-key]
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        # word_mid = (0.2 + 0.2) / 2 = 0.2; nearest speaker is SPEAKER_00.
        assert aligned["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"

    def test_speaker_embeddings_attached_when_provided(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        emb = {"SPEAKER_00": [0.1, 0.2, 0.3]}
        out = assign_word_speakers(df, result, speaker_embeddings=emb)
        assert out["speaker_embeddings"] == emb

    def test_no_embeddings_does_not_set_key(self):
        df = self._diarize_df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert "speaker_embeddings" not in out

    def test_empty_df_returns_unchanged(self):
        df = pd.DataFrame(columns=["start", "end", "speaker", "segment", "label"])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        assert out is result
        assert "speaker" not in out["segments"][0]


class TestAssignWordSpeakersMissingKeys:
    """Kills .get() default mutants by omitting start/end keys."""

    def _df(self, rows):
        return pd.DataFrame(
            [
                {
                    "segment": Segment(s, e, spk),
                    "label": 0,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                }
                for s, e, spk in rows
            ]
        )

    def test_missing_start_defaults_to_zero(self):
        # seg.get("start", 0.0): mutant changes default to None/1.0.
        # Segment without "start" -> default 0.0. Speaker at (0,1) overlaps.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"end": 1.0, "text": "hi"}],  # pyrefly: ignore[bad-typed-dict-key]
            "language": "en",
        }
        out = assign_word_speakers(df, result)
        # Correct: seg_start=0.0, overlaps (0,1) -> speaker assigned.
        # Mutant (None): seg_start=None, tree.query(None, 1.0) crashes or
        # returns nothing -> no speaker.
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_missing_end_defaults_to_zero(self):
        # seg.get("end", 0.0): mutant changes default to None/1.0.
        # Segment without "end" -> default 0.0. With start=0, query(0, 0)
        # has zero intersection. Use fill_nearest to get speaker.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"start": 0.0, "text": "hi"}],  # pyrefly: ignore[bad-typed-dict-key]
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        # Correct: seg_end=0.0, query(0, 0) empty, fill_nearest finds speaker.
        # Mutant (1.0): query(0,1) overlaps -> different path but still speaker.
        # Mutant (None): crashes.
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_missing_start_and_end_both_default_zero(self):
        # Both missing -> (0.0, 0.0). fill_nearest gets speaker at 0.0.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [{"text": "hi"}],  # pyrefly: ignore[bad-typed-dict-key]
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_segments_key_missing_defaults_to_empty_list(self):
        # transcript_result.get("segments", []): mutant changes default.
        # Result without "segments" key -> default []. No crash, returns result.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result = {"language": "en"}  # type: ignore[assignment]
        out = assign_word_speakers(df, result)  # type: ignore[arg-type]
        # Correct: segments=[] -> early return (not transcript_segments or ...).
        # Mutant (None): `not None` is True -> still early return. Equivalent.
        # Mutant (no default): crashes with KeyError. Killed by no crash.
        assert out is result

    def test_word_missing_end_defaults_to_start(self):
        # word.get("end", word_start): mutant changes default.
        # Word with start but no end -> end=start. query(start, start) empty.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: TranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hi",
                    "words": [{"word": "hi", "start": 0.5}],  # pyrefly: ignore[bad-typed-dict-key]
                }
            ],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        # Correct: word_end=0.5 (word_start). query(0.5, 0.5) empty,
        # fill_nearest finds SPEAKER_00 at 0.5.
        assert out["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"  # pyrefly: ignore[bad-typed-dict-key]


class TestAssignWordSpeakersMathKillers:
    """Kill math mutants in seg_mid and word_mid calculations.

    seg_mid = (seg_start + seg_end) / 2. Mutants: *2, -, /3. To kill, set up
    a fill_nearest scenario with two speakers where the midpoint selects one.
    A math mutant shifts the midpoint, selecting the other speaker.
    """

    def _df(self, rows):
        return pd.DataFrame(
            [
                {
                    "segment": Segment(s, e, spk),
                    "label": int(spk.split("_")[1]) if spk and "_" in spk else 0,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                }
                for s, e, spk in rows
            ]
        )

    def test_seg_mid_average_selects_closer_speaker(self):
        # seg_mid = (start+end)/2. seg (2,4) -> mid=3. A mid=0.5 (dist 2.5),
        # B mid=4.5 (dist 1.5). Correct picks B. Mutant (/3): mid=2 -> picks A.
        df = self._df([(0.0, 1.0, "SPEAKER_00"), (4.0, 5.0, "SPEAKER_01")])
        result: TranscriptionResult = {
            "segments": [{"start": 2.0, "end": 4.0, "text": "x"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        # Correct mid=3.0: B at 4.5 (dist 1.5) beats A at 0.5 (dist 2.5).
        assert out["segments"][0]["speaker"] == "SPEAKER_01"

    def test_seg_mid_subtraction_mutant_killer(self):
        # mutmut_65: (start - end) / 2. seg (2,4): correct mid=3, mutant mid=-1.
        # A at 0.5 (correct dist 2.5, mutant dist 1.5), B at 4.5 (correct 1.5,
        # mutant 5.5). Correct picks B; mutant picks A.
        df = self._df([(0.0, 1.0, "SPEAKER_00"), (4.0, 5.0, "SPEAKER_01")])
        result: TranscriptionResult = {
            "segments": [{"start": 2.0, "end": 4.0, "text": "x"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        assert out["segments"][0]["speaker"] == "SPEAKER_01"

    def test_seg_mid_multiply_mutant_killer(self):
        # mutmut_64: (start + end) * 2. seg (1,3): correct mid=2, mutant mid=8.
        # A at 0.5, B at 3.5. Correct: A dist 1.5, B dist 1.5 -> tie (A first).
        # Mutant: A dist 7.5, B dist 4.5 -> B.
        df = self._df([(0.0, 1.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")])
        result: TranscriptionResult = {
            "segments": [{"start": 1.0, "end": 3.0, "text": "x"}],
            "language": "en",
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        # With a tie, argmin returns the first (SPEAKER_00). Mutant (*2) flips
        # to SPEAKER_01.
        assert out["segments"][0]["speaker"] == "SPEAKER_00"

    def test_word_mid_average_selects_closer_speaker(self):
        # mutmut_118/119/120: word_mid math. Word (2,4): correct mid=3.
        # A mid=0.5 (dist 2.5), B mid=4.5 (dist 1.5). Correct picks B.
        df = self._df([(0.0, 1.0, "SPEAKER_00"), (4.0, 5.0, "SPEAKER_01")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "x",
                    "words": [{"word": "x", "start": 2.0, "end": 4.0, "score": 1.0}],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        assert aligned["segments"][0]["words"][0]["speaker"] == "SPEAKER_01"

    def test_word_mid_subtraction_mutant_killer(self):
        # mutmut_119: (start - end) / 2. Word (2,4): correct mid=3, mutant -1.
        df = self._df([(0.0, 1.0, "SPEAKER_00"), (4.0, 5.0, "SPEAKER_01")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 2.0,
                    "end": 4.0,
                    "text": "x",
                    "words": [{"word": "x", "start": 2.0, "end": 4.0, "score": 1.0}],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        # Correct mid=3 picks B; mutant mid=-1 picks A.
        assert aligned["segments"][0]["words"][0]["speaker"] == "SPEAKER_01"

    def test_word_mid_multiply_mutant_killer(self):
        # mutmut_118: (start + end) * 2. Word (1,3): correct mid=2, mutant 8.
        df = self._df([(0.0, 1.0, "SPEAKER_00"), (3.0, 4.0, "SPEAKER_01")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 1.0,
                    "end": 3.0,
                    "text": "x",
                    "words": [{"word": "x", "start": 1.0, "end": 3.0, "score": 1.0}],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result, fill_nearest=True)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        # Tie at mid=2 (A dist 1.5, B dist 1.5) -> first (A). Mutant mid=8 -> B.
        assert aligned["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"


class TestAssignWordSpeakersBreakKillers:
    """Kill continue->break mutant (mutmut_83) in the word loop.

    mutmut_83: continue -> break. When a word has no "start", correct skips
    it (continue) and processes subsequent words. Mutant stops the whole
    word loop (break), leaving later words unassigned.
    """

    def _df(self, rows):
        return pd.DataFrame(
            [
                {
                    "segment": Segment(s, e, spk),
                    "label": int(spk.split("_")[1]) if spk and "_" in spk else 0,
                    "speaker": spk,
                    "start": s,
                    "end": e,
                }
                for s, e, spk in rows
            ]
        )

    def test_word_without_start_does_not_stop_loop(self):
        # First word has no "start" (continue), second word has start and
        # should be assigned. Mutant (break): second word unassigned.
        df = self._df([(0.0, 1.0, "SPEAKER_00")])
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "x y",
                    "words": [
                        {"word": "x", "score": 1.0},  # pyrefly: ignore[bad-typed-dict-key]
                        {"word": "y", "start": 0.0, "end": 1.0, "score": 1.0},
                    ],
                    "chars": None,
                }
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        words = aligned["segments"][0]["words"]
        # Correct: second word assigned. Mutant (break): not assigned.
        assert words[1].get("speaker") == "SPEAKER_00"
