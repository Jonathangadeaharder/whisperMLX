"""Integration test: VAD + ASR pipeline on synthetic audio.

Mocks mlx_whisper.transcribe (volatile: model download + GPU inference) to
return a fixed segment. Verifies the VAD boundary computation and segment
assembly (offset application, language detection, progress callback).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from whisperx.asr import MlxWhisperPipeline
from whisperx.vads.vad import Vad


class _StubVad(Vad):
    """A Vad subclass whose __call__ returns one speech segment 1s-2s."""

    def __init__(self):
        super().__init__(0.5)

    def preprocess_audio(self, audio):
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
        return [{"start": 1.0, "end": 2.0, "segments": [(1.0, 2.0)]}]

    def __call__(self, audio):
        return [MagicMock(start=1.0, end=2.0, speaker="UNKNOWN")]


class TestVadAsrIntegration:
    def test_vad_boundaries_propagate_to_asr_offset(self, monkeypatch):
        # 3s of audio so the VAD segment (1s-2s) falls within bounds.
        audio = np.zeros(16000 * 3, dtype=np.float32)
        pipe = MlxWhisperPipeline(
            model_path="mlx-community/whisper-small",
            vad=_StubVad(),
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
        )
        captured_audio: list = []

        def fake_transcribe(audio_slice, **kwargs):
            captured_audio.append(np.array(audio_slice))
            return {
                "language": "en",
                "segments": [
                    {"text": "hello", "start": 0.0, "end": 0.5, "avg_logprob": -0.1},
                ],
            }

        monkeypatch.setattr("whisperx.asr.mlx_whisper.transcribe", fake_transcribe)
        result = pipe.transcribe(audio, chunk_size=30)

        # The audio slice passed to mlx_whisper spans the VAD segment (1s-2s).
        assert len(captured_audio) == 1
        slice_len = captured_audio[0].shape[0]
        assert slice_len == pytest.approx(16000, rel=0.01)  # 1s of audio
        # Segment start offset = 1.0 (VAD segment start).
        assert result["segments"][0]["start"] == 1.0
        assert result["segments"][0]["end"] == 1.5

    def test_language_detected_from_first_segment(self, monkeypatch, sine_wave_audio):
        pipe = MlxWhisperPipeline(
            model_path="mlx-community/whisper-small",
            vad=_StubVad(),
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
        )
        monkeypatch.setattr(
            "whisperx.asr.mlx_whisper.transcribe",
            lambda audio_slice, **k: {
                "language": "de",
                "segments": [{"text": "hallo", "start": 0.0, "end": 0.5}],
            },
        )
        result = pipe.transcribe(sine_wave_audio)
        assert result["language"] == "de"

    def test_multiple_vad_segments_transcribed_separately(self, monkeypatch):
        class _MultiVad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [
                    {"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]},
                    {"start": 2.0, "end": 3.0, "segments": [(2.0, 3.0)]},
                ]

            def __call__(self, audio):
                return [MagicMock(start=0.0, end=3.0, speaker="UNKNOWN")]

        pipe = MlxWhisperPipeline(
            model_path="p",
            vad=_MultiVad(),
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
        )
        call_count = {"n": 0}

        def fake_transcribe(audio_slice, **kwargs):
            call_count["n"] += 1
            return {
                "language": "en",
                "segments": [
                    {"text": f"seg{call_count['n']}", "start": 0.0, "end": 0.5},
                ],
            }

        monkeypatch.setattr("whisperx.asr.mlx_whisper.transcribe", fake_transcribe)
        audio = np.zeros(16000 * 4, dtype=np.float32)
        result = pipe.transcribe(audio, chunk_size=30)
        assert call_count["n"] == 2
        assert len(result["segments"]) == 2
        # Second segment offset by 2.0s.
        assert result["segments"][1]["start"] == 2.0

    def test_verbose_prints_transcript(self, monkeypatch, capsys, sine_wave_audio):
        pipe = MlxWhisperPipeline(
            model_path="p",
            vad=_StubVad(),
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
        )
        monkeypatch.setattr(
            "whisperx.asr.mlx_whisper.transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "hi", "start": 0.0, "end": 0.5}],
            },
        )
        pipe.transcribe(sine_wave_audio, verbose=True)
        captured = capsys.readouterr()
        assert "Transcript" in captured.out
