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


class TestAssignWordSpeakers:
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
