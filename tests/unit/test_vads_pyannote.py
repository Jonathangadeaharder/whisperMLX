"""Unit tests for whisperx.vads.pyannote (_Binarize, Pyannote VAD).

The MLX segmentation model is mocked; the numpy Binarize hysteresis logic and
the Pyannote wrapper wiring are the behavior under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from whisperx.vads.pyannote import Pyannote, _Binarize, _SpeechSegment


class TestSpeechSegment:
    def test_attributes(self):
        seg = _SpeechSegment(1.0, 3.0)
        assert seg.start == 1.0
        assert seg.end == 3.0
        assert seg.duration == 2.0


class TestBinarizeInit:
    def test_defaults(self):
        b = _Binarize()
        assert b.onset == 0.5
        assert b.offset == 0.5  # defaults to onset
        assert b.min_duration_on == 0.0
        assert b.min_duration_off == 0.0

    def test_offset_defaults_to_onset(self):
        b = _Binarize(onset=0.7)
        assert b.offset == 0.7

    def test_custom_params(self):
        b = _Binarize(onset=0.6, offset=0.3, min_duration_on=0.2, min_duration_off=0.5)
        assert b.onset == 0.6
        assert b.offset == 0.3
        assert b.min_duration_on == 0.2
        assert b.min_duration_off == 0.5


class TestBinarizeCall:
    def _frames(self, scores, dt=0.1):
        return np.arange(len(scores)) * dt, np.array(scores, dtype=np.float32)

    def test_all_below_onset_returns_empty(self):
        b = _Binarize(onset=0.5, offset=0.3)
        times, scores = self._frames([0.1, 0.2, 0.1])
        segs = b(scores, times)
        assert segs == []

    def test_constant_high_returns_single_segment(self):
        b = _Binarize(onset=0.5, offset=0.3)
        times, scores = self._frames([0.9, 0.9, 0.9, 0.9])
        segs = b(scores, times)
        assert len(segs) == 1
        assert segs[0].start == 0.0
        assert segs[0].end == pytest.approx(0.3)

    def test_silence_closes_segment(self):
        b = _Binarize(onset=0.5, offset=0.3)
        times, scores = self._frames([0.9, 0.9, 0.1, 0.1])
        segs = b(scores, times)
        assert len(segs) == 1
        assert segs[0].start == 0.0
        # closes at first below-offset frame (t=0.2)
        assert segs[0].end == pytest.approx(0.2)

    def test_multiple_segments(self):
        b = _Binarize(onset=0.5, offset=0.3)
        times, scores = self._frames([0.9, 0.1, 0.9, 0.1])
        segs = b(scores, times)
        assert len(segs) == 2

    def test_2d_scores_flattened(self):
        b = _Binarize(onset=0.5, offset=0.3)
        times = np.array([0.0, 0.1, 0.2])
        scores = np.array([[0.9], [0.9], [0.1]], dtype=np.float32)
        segs = b(scores, times)
        assert len(segs) == 1

    def test_max_duration_splits_long_run(self):
        # A long high run exceeding max_duration is split at its min-score point.
        b = _Binarize(onset=0.5, offset=0.3, max_duration=0.25)
        times = np.arange(10) * 0.1
        scores = np.array([0.9, 0.9, 0.6, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        segs = b(scores, times)
        # At least two segments because duration exceeds max_duration.
        assert len(segs) >= 2

    def test_min_duration_on_filters_short(self):
        b = _Binarize(onset=0.5, offset=0.3, min_duration_on=0.5)
        times = np.arange(4) * 0.1
        # 2-frame high run (0.2s duration) -> below 0.5s threshold -> removed.
        scores = np.array([0.9, 0.9, 0.1, 0.1], dtype=np.float32)
        segs = b(scores, times)
        assert segs == []

    def test_min_duration_off_merges_close_segments(self):
        b = _Binarize(onset=0.5, offset=0.3, min_duration_off=0.5)
        # Two segments with a 0.1s gap (below min_duration_off=0.5) -> merged.
        times = np.arange(6) * 0.1
        scores = np.array([0.9, 0.1, 0.9, 0.1, 0.9, 0.1], dtype=np.float32)
        segs = b(scores, times)
        assert len(segs) == 1

    def test_pad_onset_offset(self):
        b = _Binarize(onset=0.5, offset=0.3, pad_onset=0.05, pad_offset=0.05)
        times = np.array([0.5, 0.6, 0.7])
        scores = np.array([0.9, 0.1, 0.1], dtype=np.float32)
        segs = b(scores, times)
        assert len(segs) == 1
        assert segs[0].start == pytest.approx(0.45)
        assert segs[0].end == pytest.approx(0.65)


class TestPyannote:
    def test_init_stores_params(self, monkeypatch):
        with patch("whisperx.mlx_models.pyannote_segmentation.segment_audio"):
            v = Pyannote(vad_onset=0.5, vad_offset=0.363, chunk_size=30)
        assert v.vad_onset == 0.5
        assert v.vad_offset == 0.363
        assert v.chunk_size == 30

    def test_call_returns_segmentx_list(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.pyannote_segmentation.segment_audio") as sa:
            # 2 frames: one high, one low -> single segment.
            sa.return_value = (
                mx_module.array([[0.9], [0.1]]),
                mx_module.array([0.0, 0.1]),
            )
            v = Pyannote(vad_onset=0.5, vad_offset=0.3, chunk_size=30)
            result = v({"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000})
        assert isinstance(result, list)
        assert len(result) == 1
        assert hasattr(result[0], "start")
        assert hasattr(result[0], "end")
        assert result[0].speaker == "UNKNOWN"

    def test_call_accepts_torch_like_waveform(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.pyannote_segmentation.segment_audio") as sa:
            sa.return_value = (mx_module.array([[0.9], [0.1]]), mx_module.array([0.0, 0.1]))
            v = Pyannote(vad_onset=0.5, vad_offset=0.3, chunk_size=30)
            waveform = MagicMock()
            waveform.numpy.return_value = np.zeros(16000, dtype=np.float32)
            result = v({"waveform": waveform, "sample_rate": 16000})
        assert len(result) == 1

    def test_call_2d_waveform_squeezed(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.pyannote_segmentation.segment_audio") as sa:
            sa.return_value = (mx_module.array([[0.9], [0.1]]), mx_module.array([0.0, 0.1]))
            v = Pyannote(vad_onset=0.5, vad_offset=0.3, chunk_size=30)
            # 2D waveform (2, N) -> squeezed to 1D.
            audio2d = np.zeros((2, 16000), dtype=np.float32)
            result = v({"waveform": audio2d, "sample_rate": 16000})
        assert len(result) == 1

    def test_preprocess_audio_is_identity(self):
        audio = object()
        assert Pyannote.preprocess_audio(audio) is audio

    def test_merge_chunks_empty_warns(self, caplog):
        with caplog.at_level("WARNING", logger="whisperx.vads.pyannote"):
            out = Pyannote.merge_chunks([], chunk_size=30)
        assert out == []

    def test_merge_chunks_passes_to_base(self, make_segments):
        segs = make_segments([(0.0, 1.0), (1.0, 2.0)])
        out = Pyannote.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert len(out) == 1
        assert out[0]["end"] == 2.0

    def test_merge_chunks_chunk_size_must_be_positive(self):
        with pytest.raises(AssertionError):
            Pyannote.merge_chunks([], chunk_size=0)
