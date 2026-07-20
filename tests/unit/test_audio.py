"""Unit tests for whisperx.audio.load_audio (ffmpeg subprocess mocked)."""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock

import numpy as np
import pytest
from whisperx import audio
from whisperx.audio import SAMPLE_RATE, load_audio


def _make_pcm_bytes(samples: np.ndarray) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    return pcm.tobytes()


class TestLoadAudio:
    def test_decodes_int16_to_float32(self, monkeypatch):
        samples = np.array([0.0, 0.5, -0.5, 1.0], dtype=np.float32)
        pcm = _make_pcm_bytes(samples)

        completed = MagicMock()
        completed.stdout = pcm
        monkeypatch.setattr(subprocess, "run", lambda *a, **k: completed, raising=False)
        monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: completed)

        out = load_audio("dummy.wav")
        assert out.dtype == np.float32
        # float32 values in [-1, 1]
        assert np.all(out <= 1.0)
        assert np.all(out >= -1.0)
        # 4 samples decoded
        assert out.shape == (4,)

    def test_uses_default_sample_rate(self, monkeypatch):
        captured: dict = {}

        completed = MagicMock()
        completed.stdout = _make_pcm_bytes(np.zeros(8, dtype=np.float32))

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return completed

        monkeypatch.setattr(audio.subprocess, "run", fake_run)
        load_audio("dummy.wav")
        cmd = captured["cmd"]
        assert cmd[0] == "ffmpeg"
        assert "-ar" in cmd
        assert str(SAMPLE_RATE) in cmd
        assert "-ac" in cmd
        assert "1" in cmd

    def test_custom_sample_rate_passed_to_ffmpeg(self, monkeypatch):
        captured: dict = {}
        completed = MagicMock()
        completed.stdout = _make_pcm_bytes(np.zeros(4, dtype=np.float32))

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            return completed

        monkeypatch.setattr(audio.subprocess, "run", fake_run)
        load_audio("dummy.wav", sr=8000)
        assert "8000" in captured["cmd"]

    def test_called_process_error_raises_runtime_error(self, monkeypatch):
        err = subprocess.CalledProcessError(returncode=1, cmd=["ffmpeg"])
        err.stderr = b"boom"

        def fake_run(cmd, **kwargs):
            raise err

        monkeypatch.setattr(audio.subprocess, "run", fake_run)
        with pytest.raises(RuntimeError, match="Failed to load audio"):
            load_audio("missing.wav")

    def test_monaural_flatten(self, monkeypatch):
        # frombuffer of interleaved mono int16 -> 1D array.
        pcm = _make_pcm_bytes(np.linspace(-1.0, 1.0, 16, dtype=np.float32))
        completed = MagicMock()
        completed.stdout = pcm
        monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: completed)
        out = load_audio("dummy.wav")
        assert out.ndim == 1
        assert out.shape[0] == 16

    def test_normalization_divides_by_32768(self, monkeypatch):
        samples = np.array([1.0, -1.0], dtype=np.float32)
        completed = MagicMock()
        completed.stdout = _make_pcm_bytes(samples)
        monkeypatch.setattr(audio.subprocess, "run", lambda *a, **k: completed)
        out = load_audio("dummy.wav")
        assert np.isclose(out[0], 32767.0 / 32768.0, atol=1e-4)
        assert np.isclose(out[1], -32767.0 / 32768.0, atol=1e-4)
