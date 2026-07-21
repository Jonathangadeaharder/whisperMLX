"""Unit tests for whisperx.__init__ lazy import wrappers.

The package exposes thin lazy-import wrappers so the heavy ASR/VAD/diarize
stack is only loaded on first use. These tests exercise each wrapper through
the public package API, mocking the underlying modules to avoid runtime costs.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest
import whisperx


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="whisperx.asr -> whisperx.vads -> mlx.core",
)
class TestLazyImportsAsr:
    def test_load_model_delegates(self):
        with patch("whisperx.asr.load_model") as fn:
            fn.return_value = "pipeline"
            out = whisperx.load_model("small", "cpu")
        fn.assert_called_once_with("small", "cpu")
        assert out == "pipeline"


class TestLazyImports:
    def test_load_align_model_delegates(self):
        with patch("whisperx.alignment.load_align_model") as fn:
            fn.return_value = ("model", {"language": "en"})
            out = whisperx.load_align_model("en", "cpu")
        fn.assert_called_once_with("en", "cpu")
        assert out[1]["language"] == "en"

    def test_align_delegates(self):
        with patch("whisperx.alignment.align") as fn:
            fn.return_value = {"segments": [], "word_segments": []}
            out = whisperx.align([], MagicMock(), {}, MagicMock(), "cpu")
        fn.assert_called_once()
        assert "segments" in out

    def test_load_audio_delegates(self):
        with patch("whisperx.audio.load_audio") as fn:
            fn.return_value = "audio"
            out = whisperx.load_audio("file.wav")
        fn.assert_called_once_with("file.wav")
        assert out == "audio"

    def test_assign_word_speakers_delegates(self):
        with patch("whisperx.diarize.assign_word_speakers") as fn:
            fn.return_value = {"segments": []}
            out = whisperx.assign_word_speakers(MagicMock(), {})
        fn.assert_called_once()
        assert "segments" in out

    def test_setup_logging_delegates(self):
        with patch("whisperx.log_utils.setup_logging") as fn:
            whisperx.setup_logging(level="info")
        fn.assert_called_once_with(level="info")

    def test_get_logger_delegates(self):
        with patch("whisperx.log_utils.get_logger") as fn:
            fn.return_value = "logger"
            out = whisperx.get_logger("whisperx.utils")
        fn.assert_called_once_with("whisperx.utils")
        assert out == "logger"
