"""Shared pytest fixtures for the whisperMLX test suite.

Pytest-only conventions: fixtures over decorators, tmp_path for temp files,
monkeypatch for env. MLX (mlx.core) is a compiled .so that pyrefly cannot
resolve, so it is imported lazily inside test functions or fixtures, never at
module top-level here.
"""

from __future__ import annotations

import contextlib
import sys
import wave
from collections.abc import Iterator

import numpy as np

# MLX is Apple Silicon only. Tests that import whisperx.asr / whisperx.vads
# pull in mlx_whisper -> mlx.core, which fails on Linux CI. Skip those
# modules at collection time on non-darwin platforms.
if sys.platform != "darwin":
    collect_ignore_glob = [
        "unit/test_asr.py",
        "unit/test_mlx_forward.py",
        "unit/test_pyannote_segmentation.py",
        "unit/test_silero_vad.py",
        "unit/test_transcribe.py",
        "unit/test_vad.py",
        "unit/test_vads_pyannote.py",
        "unit/test_vads_silero.py",
        "unit/test_wespeaker.py",
        "integration/test_diarize_components.py",
        "integration/test_vad_asr_integration.py",
    ]
import pytest

SAMPLE_RATE = 16000


def _write_wav(path: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a mono float32 waveform to a 16-bit PCM WAV file."""
    pcm = np.clip(samples, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())


@pytest.fixture
def sine_wave_audio() -> np.ndarray:
    """1 second of a 440Hz mono sine wave at 16kHz, float32 in [-1, 1]."""
    n = SAMPLE_RATE
    t = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def tmp_wav_path(tmp_path, sine_wave_audio) -> str:
    """A real WAV file on disk holding one second of 440Hz sine."""
    path = str(tmp_path / "tone.wav")
    _write_wav(path, sine_wave_audio)
    return path


@pytest.fixture
def tmp_wav_factory(tmp_path):
    """Factory: write an arbitrary waveform to a temp WAV, return its path."""

    counter = {"i": 0}

    def _make(samples: np.ndarray, name: str | None = None, sample_rate: int = SAMPLE_RATE) -> str:
        counter["i"] += 1
        fname = name or f"audio_{counter['i']}.wav"
        path = str(tmp_path / fname)
        _write_wav(path, samples, sample_rate=sample_rate)
        return path

    return _make


@pytest.fixture
def silent_audio() -> np.ndarray:
    """One second of silence at 16kHz."""
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def mx_module():
    """Lazily import mlx.core; tests use this to build deterministic arrays."""
    import mlx.core as mx  # pyrefly: ignore[missing-import]

    return mx


@pytest.fixture
def make_segments():
    """Factory for SegmentX-like objects used by Vad.merge_chunks.

    The merge_chunks helper reads .start/.end/.speaker attributes, so a small
    namespace is enough and avoids importing the diarize stack eagerly.
    """

    class _Seg:
        def __init__(self, start: float, end: float, speaker: str | None = None) -> None:
            self.start = start
            self.end = end
            self.speaker = speaker

    def _build(pairs):
        return [_Seg(s, e, "UNKNOWN") for s, e in pairs]

    return _build


@pytest.fixture
def reset_logger_handlers() -> Iterator[None]:
    """Clear the 'whisperx' logger handlers around a test so log setup is clean.

    Closes any handlers opened during the test (e.g. FileHandler) to avoid
    ResourceWarning under filterwarnings=error.
    """
    import logging

    logger = logging.getLogger("whisperx")
    saved_handlers = logger.handlers[:]
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    yield
    for h in logger.handlers:
        with contextlib.suppress(Exception):
            h.close()
    logger.handlers.clear()
    for h in saved_handlers:
        logger.addHandler(h)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


@pytest.fixture(autouse=True)
def _np_seed():
    """Deterministic numpy RNG per test for reproducible synthetic data."""
    np.random.seed(0)
