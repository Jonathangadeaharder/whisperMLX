"""Unit tests for whisperx.transcribe.transcribe_task orchestration.

All sub-pipelines (ASR load_model + transcribe, align load_align_model + align,
DiarizationPipeline, load_audio, get_writer) are mocked; transcribe_task's
argument handling, language validation, and wiring are the behavior under test.
"""

from __future__ import annotations

import argparse
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from whisperx import transcribe as tr


def _base_args(**overrides):
    """Return a complete args dict matching the CLI parser in __main__.py."""
    args = {
        "model": "small",
        "batch_size": 8,
        "model_dir": None,
        "model_cache_only": False,
        "output_dir": ".",
        "output_format": "json",
        "device": "cpu",
        "device_index": 0,
        "compute_type": "default",
        "verbose": False,
        "align_model": None,
        "interpolate_method": "nearest",
        "no_align": False,
        "task": "transcribe",
        "return_char_alignments": False,
        "hf_token": None,
        "vad_method": "pyannote",
        "vad_onset": 0.5,
        "vad_offset": 0.363,
        "chunk_size": 30,
        "diarize": False,
        "min_speakers": None,
        "max_speakers": None,
        "diarize_model": "pyannote/speaker-diarization-3.1",
        "print_progress": False,
        "speaker_embeddings": False,
        "language": None,
        "temperature": 0.0,
        "temperature_increment_on_fallback": 0.2,
        "threads": 0,
        "beam_size": 5,
        "patience": 1.0,
        "length_penalty": 1.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "initial_prompt": None,
        "hotwords": None,
        "suppress_tokens": "-1",
        "suppress_numerals": False,
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
        "audio": ["dummy.wav"],
    }
    args.update(overrides)
    return args


def _make_parser():
    return argparse.ArgumentParser()


@pytest.fixture
def mock_asr_pipeline():
    pipe = MagicMock()
    pipe.transcribe.return_value = {
        "segments": [{"start": 0.0, "end": 1.0, "text": "hello world"}],
        "language": "en",
    }
    return pipe


@pytest.fixture
def _patch_pipelines(mock_asr_pipeline, tmp_path):
    """Patch all heavy sub-pipelines; writer writes to a tmp dir."""
    audio = np.zeros(16000, dtype=np.float32)
    writer = MagicMock()
    contexts = []

    def _enter():
        patches = [
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=audio),
            patch("whisperx.transcribe.load_align_model"),
            patch("whisperx.transcribe.align"),
            patch("whisperx.transcribe.DiarizationPipeline"),
            patch("whisperx.transcribe.get_writer", return_value=writer),
            patch("whisperx.transcribe.os.makedirs"),
        ]
        for p in patches:
            p.start()
            contexts.append(p)
        return writer

    def _exit():
        for p in contexts:
            p.stop()

    return _enter, _exit


class TestTranscribeTaskBasics:
    def test_runs_asr_and_writes_output(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=True)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer") as gw,
        ):
            writer = MagicMock()
            gw.return_value = writer
            tr.transcribe_task(args, _make_parser())
        mock_asr_pipeline.transcribe.assert_called_once()
        writer.assert_called_once()
        # The writer is called with the result dict and audio path.
        call_args = writer.call_args
        assert call_args[0][1] == "dummy.wav"

    def test_skips_align_when_no_align(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=True)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.load_align_model") as la,
        ):
            tr.transcribe_task(args, _make_parser())
        la.assert_not_called()

    def test_translate_task_forces_no_align(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), task="translate")
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.load_align_model") as la,
        ):
            tr.transcribe_task(args, _make_parser())
        la.assert_not_called()


class TestLanguageValidation:
    def test_language_name_resolves_to_code(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=True, language="English")
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline) as lm,
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
        ):
            tr.transcribe_task(args, _make_parser())
        # load_model called with language="en"
        assert lm.call_args.kwargs["language"] == "en"

    def test_unsupported_language_raises(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=True, language="klingon")
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
        ):
            with pytest.raises(ValueError, match="Unsupported language"):
                tr.transcribe_task(args, _make_parser())

    def test_en_model_forces_english_language(self, mock_asr_pipeline, tmp_path):
        args = _base_args(
            output_dir=str(tmp_path),
            no_align=True,
            model="small.en",
            language="fr",
        )
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline) as lm,
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            pytest.warns(UserWarning, match="English-only"),
        ):
            tr.transcribe_task(args, _make_parser())
        assert lm.call_args.kwargs["language"] == "en"


class TestAlignmentWiring:
    def test_align_runs_when_segments_present(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=False)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.load_align_model") as la,
            patch("whisperx.transcribe.align") as al,
        ):
            la.return_value = (
                MagicMock(),
                {"language": "en", "dictionary": {"a": 1}, "type": "hf"},
            )
            al.return_value = {
                "segments": [{"start": 0.0, "end": 1.0, "text": "hi", "words": []}],
                "word_segments": [],
            }
            tr.transcribe_task(args, _make_parser())
        la.assert_called_once()
        al.assert_called_once()

    def test_align_skipped_when_no_segments(self, mock_asr_pipeline, tmp_path):
        mock_asr_pipeline.transcribe.return_value = {"segments": [], "language": "en"}
        args = _base_args(output_dir=str(tmp_path), no_align=False)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.load_align_model") as la,
            patch("whisperx.transcribe.align") as al,
        ):
            la.return_value = (MagicMock(), {"language": "en", "dictionary": {}, "type": "hf"})
            tr.transcribe_task(args, _make_parser())
        # align not called because no segments.
        al.assert_not_called()

    def test_new_language_triggers_align_model_reload(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=False)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.load_align_model") as la,
            patch("whisperx.transcribe.align") as al,
        ):
            # First call returns en metadata; result language is fr -> reload.
            la.side_effect = [
                (MagicMock(), {"language": "en", "dictionary": {}, "type": "hf"}),
                (MagicMock(), {"language": "fr", "dictionary": {}, "type": "hf"}),
            ]
            mock_asr_pipeline.transcribe.return_value = {
                "segments": [{"start": 0.0, "end": 1.0, "text": "bonjour"}],
                "language": "fr",
            }
            al.return_value = {"segments": [], "word_segments": []}
            tr.transcribe_task(args, _make_parser())
        assert la.call_count == 2


class TestDiarizeWiring:
    def test_diarize_runs_when_enabled(self, mock_asr_pipeline, tmp_path):
        args = _base_args(output_dir=str(tmp_path), no_align=True, diarize=True)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.DiarizationPipeline") as DP,
            patch("whisperx.transcribe.assign_word_speakers") as aws,
        ):
            pipe = MagicMock()
            pipe.return_value = pd.DataFrame(
                [{"segment": None, "label": 0, "speaker": "SPEAKER_00", "start": 0.0, "end": 1.0}]
            )
            DP.return_value = pipe
            tr.transcribe_task(args, _make_parser())
        DP.assert_called_once()
        pipe.assert_called_once()
        aws.assert_called_once()

    def test_speaker_embeddings_without_diarize_warns(self, mock_asr_pipeline, tmp_path):
        args = _base_args(
            output_dir=str(tmp_path), no_align=True, diarize=False, speaker_embeddings=True
        )
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
        ):
            with pytest.warns(UserWarning, match="no effect without --diarize"):
                tr.transcribe_task(args, _make_parser())

    def test_diarize_returns_embeddings(self, mock_asr_pipeline, tmp_path):
        args = _base_args(
            output_dir=str(tmp_path),
            no_align=True,
            diarize=True,
            speaker_embeddings=True,
        )
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.DiarizationPipeline") as DP,
            patch("whisperx.transcribe.assign_word_speakers") as aws,
        ):
            pipe = MagicMock()
            pipe.return_value = (
                pd.DataFrame(
                    [
                        {
                            "speaker": "SPEAKER_00",
                            "start": 0.0,
                            "end": 1.0,
                            "label": 0,
                            "segment": None,
                        }
                    ]
                ),
                {"SPEAKER_00": [0.1, 0.2]},
            )
            DP.return_value = pipe
            tr.transcribe_task(args, _make_parser())
        # assign_word_speakers called with the embeddings dict.
        assert aws.call_args.args[2] == {"SPEAKER_00": [0.1, 0.2]}


class TestWordOptionsValidation:
    def test_word_option_with_no_align_errors(self, mock_asr_pipeline, tmp_path):
        parser = _make_parser()
        parser.error = MagicMock(side_effect=SystemExit(2))
        args = _base_args(output_dir=str(tmp_path), no_align=True, highlight_words=True)
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
        ):
            with pytest.raises(SystemExit):
                tr.transcribe_task(args, parser)
        parser.error.assert_called_once()

    def test_max_line_count_without_width_warns(self, mock_asr_pipeline, tmp_path):
        # With alignment enabled, max_line_count without max_line_width warns.
        args = _base_args(
            output_dir=str(tmp_path),
            no_align=False,
            max_line_count=3,
            max_line_width=None,
        )
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline),
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
            patch("whisperx.transcribe.load_align_model") as la,
            patch("whisperx.transcribe.align") as al,
        ):
            la.return_value = (MagicMock(), {"language": "en", "dictionary": {}, "type": "hf"})
            al.return_value = {"segments": [], "word_segments": []}
            with pytest.warns(UserWarning, match="no effect without --max_line_width"):
                tr.transcribe_task(args, _make_parser())


class TestTemperatureHandling:
    def test_temperature_increment_builds_tuple(self, mock_asr_pipeline, tmp_path):
        args = _base_args(
            output_dir=str(tmp_path),
            no_align=True,
            temperature=0.0,
            temperature_increment_on_fallback=0.2,
        )
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline) as lm,
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
        ):
            tr.transcribe_task(args, _make_parser())
        temps = lm.call_args.kwargs["asr_options"]["temperatures"]
        # A tuple of np.arange values.
        assert hasattr(temps, "__iter__")
        assert len(temps) > 1

    def test_no_increment_keeps_list(self, mock_asr_pipeline, tmp_path):
        args = _base_args(
            output_dir=str(tmp_path),
            no_align=True,
            temperature=0.5,
            temperature_increment_on_fallback=None,
        )
        with (
            patch("whisperx.transcribe.load_model", return_value=mock_asr_pipeline) as lm,
            patch("whisperx.transcribe.load_audio", return_value=np.zeros(16000, dtype=np.float32)),
            patch("whisperx.transcribe.os.makedirs"),
            patch("whisperx.transcribe.get_writer"),
        ):
            tr.transcribe_task(args, _make_parser())
        temps = lm.call_args.kwargs["asr_options"]["temperatures"]
        assert list(temps) == [0.5]
