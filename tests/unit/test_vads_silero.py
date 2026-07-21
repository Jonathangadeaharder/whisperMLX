"""Unit tests for whisperx.vads.silero (Silero VAD wrapper).

The MLX Silero detect_speech function is mocked; the wrapper wiring, sample
rate validation, and merge_chunks logic are the behavior under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from whisperx.vads.silero import Silero


class TestSileroInit:
    def test_stores_params(self, monkeypatch):
        with patch("whisperx.mlx_models.silero_vad.detect_speech"):
            v = Silero(vad_onset=0.5, chunk_size=30)
        assert v.vad_onset == 0.5
        assert v.chunk_size == 30


class TestSileroCall:
    def test_returns_segmentx_list(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.silero_vad.detect_speech") as ds:
            ds.return_value = [(0, 16000), (32000, 48000)]
            v = Silero(vad_onset=0.5, chunk_size=30)
            result = v({"waveform": np.zeros(48000, dtype=np.float32), "sample_rate": 16000})
        assert len(result) == 2
        assert hasattr(result[0], "start")
        assert hasattr(result[0], "end")
        assert result[0].speaker == "UNKNOWN"
        # Sample times converted from samples to seconds (16000 samples = 1s).
        assert result[0].start == 0.0
        assert result[0].end == 1.0

    def test_rejects_non_16k_sample_rate(self, monkeypatch):
        with patch("whisperx.mlx_models.silero_vad.detect_speech"):
            v = Silero(vad_onset=0.5, chunk_size=30)
            with pytest.raises(ValueError, match="Only 16000Hz"):
                v({"waveform": np.zeros(1000, dtype=np.float32), "sample_rate": 8000})

    def test_accepts_torch_like_waveform(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.silero_vad.detect_speech") as ds:
            ds.return_value = [(0, 16000)]
            v = Silero(vad_onset=0.5, chunk_size=30)
            waveform = MagicMock()
            waveform.numpy.return_value = np.zeros(16000, dtype=np.float32)
            result = v({"waveform": waveform, "sample_rate": 16000})
        assert len(result) == 1

    def test_2d_waveform_squeezed(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.silero_vad.detect_speech") as ds:
            ds.return_value = [(0, 16000)]
            v = Silero(vad_onset=0.5, chunk_size=30)
            audio2d = np.zeros((2, 16000), dtype=np.float32)
            result = v({"waveform": audio2d, "sample_rate": 16000})
        assert len(result) == 1

    def test_passes_threshold_and_chunk_size(self, monkeypatch, mx_module):
        with patch("whisperx.mlx_models.silero_vad.detect_speech") as ds:
            ds.return_value = []
            v = Silero(vad_onset=0.7, chunk_size=15)
            v({"waveform": np.zeros(16000, dtype=np.float32), "sample_rate": 16000})
        ds.assert_called_once()
        _, kwargs = ds.call_args
        assert kwargs["threshold"] == 0.7
        assert kwargs["chunk_size"] == 512
        assert kwargs["max_speech_duration_s"] == 15


class TestSileroHelpers:
    def test_preprocess_audio_is_identity(self):
        audio = object()
        assert Silero.preprocess_audio(audio) is audio

    def test_merge_chunks_empty_warns(self, caplog):
        with caplog.at_level("WARNING", logger="whisperx.vads.silero"):
            out = Silero.merge_chunks([], chunk_size=30)
        assert out == []

    def test_merge_chunks_passes_to_base(self, make_segments):
        segs = make_segments([(0.0, 1.0), (1.0, 2.0)])
        out = Silero.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert len(out) == 1

    def test_merge_chunks_requires_positive_chunk_size(self):
        with pytest.raises(AssertionError):
            Silero.merge_chunks([], chunk_size=-1)
