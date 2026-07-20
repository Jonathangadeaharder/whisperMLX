"""Integration test: diarization components wired together.

Mocks the MLX segmentation + WeSpeaker embedder (volatile: MLX runtime +
model download) and verifies the full clustering -> assign_word_speakers flow
assigns consistent speaker labels to words.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from whisperx.diarize import DiarizationPipeline, assign_word_speakers
from whisperx.schema import AlignedTranscriptionResult


def _embeddings_factory(speakers):
    """Return an embed function that yields a distinct vector per speaker."""
    state = {"i": 0}

    def _embed(audio, weights=None):
        spk = speakers[state["i"] % len(speakers)]
        state["i"] += 1
        v = np.zeros(8, dtype=np.float32)
        v[spk] = 1.0
        return v

    return _embed


class TestDiarizeClusteringIntegration:
    def test_two_speakers_clustered_consistently(self, monkeypatch):
        # Two well-separated speakers; embeddings alternate.
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe._segment_audio = MagicMock(
            return_value=(np.array([[0.9], [0.9], [0.9], [0.9]]), np.array([0.5, 1.0, 1.5, 2.0]))
        )
        pipe._embed = MagicMock(side_effect=_embeddings_factory([0, 1, 0, 1]))
        pipe._wespeaker_weights = {}
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363
        monkeypatch.setattr(
            DiarizationPipeline,
            "_binarize_segments",
            lambda self, scores, frame_times: [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        )
        out = pipe(np.zeros(16000 * 4, dtype=np.float32), num_speakers=2)
        assert len(out) == 4
        speakers = sorted(out["speaker"].unique())
        assert len(speakers) == 2
        # Speaker labels are formatted SPEAKER_NN.
        for spk in speakers:
            assert spk.startswith("SPEAKER_")

    def test_embeddings_returned_with_mean_per_speaker(self, monkeypatch):
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe._segment_audio = MagicMock(
            return_value=(np.array([[0.9], [0.9], [0.9], [0.9]]), np.array([0.5, 1.0, 1.5, 2.0]))
        )
        pipe._embed = MagicMock(side_effect=_embeddings_factory([0, 1, 0, 1]))
        pipe._wespeaker_weights = {}
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363
        monkeypatch.setattr(
            DiarizationPipeline,
            "_binarize_segments",
            lambda self, scores, frame_times: [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0)],
        )
        result = pipe(np.zeros(16000 * 4, dtype=np.float32), num_speakers=2, return_embeddings=True)
        assert isinstance(result, tuple)
        df, embs = result
        assert embs is not None
        assert set(embs.keys()) == set(df["speaker"].unique())
        for _spk, emb in embs.items():
            assert len(emb) == 8

    def test_clustering_with_min_max_speakers(self, monkeypatch):
        pipe = DiarizationPipeline.__new__(DiarizationPipeline)
        pipe._segment_audio = MagicMock(return_value=(np.array([[0.9]] * 6), np.arange(6) * 0.5))
        pipe._embed = MagicMock(side_effect=_embeddings_factory([0, 1, 0, 1, 0, 1]))
        pipe._wespeaker_weights = {}
        pipe.vad_onset = 0.5
        pipe.vad_offset = 0.363
        monkeypatch.setattr(
            DiarizationPipeline,
            "_binarize_segments",
            lambda self, scores, frame_times: [(i * 1.0, i * 1.0 + 0.5) for i in range(6)],
        )
        out = pipe(np.zeros(16000 * 6, dtype=np.float32), min_speakers=1, max_speakers=3)
        assert isinstance(out, pd.DataFrame)
        n_speakers = out["speaker"].nunique()
        assert 1 <= n_speakers <= 3


class TestAssignWordSpeakersIntegration:
    def test_full_assignment_flow(self):
        # Diarize dataframe with two non-overlapping speakers.
        from whisperx.diarize import Segment

        df = pd.DataFrame(
            [
                {
                    "segment": Segment(0.0, 1.0, "SPEAKER_00"),
                    "label": 0,
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.0,
                },
                {
                    "segment": Segment(1.0, 2.0, "SPEAKER_01"),
                    "label": 1,
                    "speaker": "SPEAKER_01",
                    "start": 1.0,
                    "end": 2.0,
                },
            ]
        )
        result: AlignedTranscriptionResult = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.4, "score": 1.0},
                        {"word": "world", "start": 0.5, "end": 0.9, "score": 1.0},
                    ],
                    "chars": None,
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "text": "foo bar",
                    "words": [
                        {"word": "foo", "start": 1.1, "end": 1.4, "score": 1.0},
                        {"word": "bar", "start": 1.5, "end": 1.9, "score": 1.0},
                    ],
                    "chars": None,
                },
            ],
            "word_segments": [],
        }
        out = assign_word_speakers(df, result)
        aligned: AlignedTranscriptionResult = out  # pyrefly: ignore[bad-assignment]
        assert out["segments"][0]["speaker"] == "SPEAKER_00"
        assert out["segments"][1]["speaker"] == "SPEAKER_01"
        assert aligned["segments"][0]["words"][0]["speaker"] == "SPEAKER_00"
        assert aligned["segments"][1]["words"][1]["speaker"] == "SPEAKER_01"
