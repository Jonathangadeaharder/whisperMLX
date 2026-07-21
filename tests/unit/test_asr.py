"""Unit tests for whisperx.asr (MlxWhisperPipeline, load_model, helpers).

mlx_whisper.transcribe (volatile: model download + GPU inference) and VAD
construction are mocked. The segment-assembly logic is the behavior under test.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from whisperx import asr as asr_mod
from whisperx.asr import (
    MlxWhisperPipeline,
    _resolve_model_path,
    find_numeral_symbol_tokens,
    load_model,
)


class TestResolveModelPath:
    def test_alias_resolves_to_mlx_community(self):
        assert _resolve_model_path("small") == "mlx-community/whisper-small"

    def test_slash_path_passes_through(self):
        assert _resolve_model_path("org/repo") == "org/repo"

    def test_mlx_prefix_passes_through(self):
        assert (
            _resolve_model_path("mlx-community/whisper-large-v3")
            == "mlx-community/whisper-large-v3"
        )

    def test_local_path_passes_through(self):
        assert _resolve_model_path("/models/whisper-small") == "/models/whisper-small"


class TestFindNumeralSymbolTokens:
    def test_finds_digit_tokens(self):
        tokenizer = MagicMock()
        tokenizer.eot = 20
        # decode(i) returns " "+token string; digits at indices 5 and 10.
        decode_map = {i: f" word{i}" for i in range(20)}
        decode_map[5] = " 42"
        decode_map[10] = " $5"
        tokenizer.decode.side_effect = lambda ids: decode_map.get(ids[0], "x")
        tokens = find_numeral_symbol_tokens(tokenizer)
        assert 5 in tokens
        assert 10 in tokens

    def test_no_eot_uses_vocab_size(self):
        tokenizer = MagicMock()
        tokenizer.eot = None
        tokenizer.vocab_size = 5
        tokenizer.decode.side_effect = lambda ids: "no digits"
        tokens = find_numeral_symbol_tokens(tokenizer)
        assert tokens == []

    def test_strips_leading_space(self):
        tokenizer = MagicMock()
        tokenizer.eot = 3
        # decode returns " 5" (with leading space); the '5' should be detected.
        tokenizer.decode.side_effect = lambda ids: {0: " a", 1: " 5", 2: " z"}[ids[0]]
        tokens = find_numeral_symbol_tokens(tokenizer)
        assert 1 in tokens
        assert 0 not in tokens


def _stub_vad():
    """A minimal VAD-like object that is an instance of Vad subclass."""
    from whisperx.vads.vad import Vad

    class _StubVad(Vad):
        def __init__(self):
            super().__init__(0.5)

        def preprocess_audio(self, audio):
            return audio

        @staticmethod
        def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
            return Vad.merge_chunks(segments, chunk_size, onset, offset)

        def __call__(self, audio):
            return []

    return _StubVad()


def _chunked_vad(chunks):
    """Vad subclass returning fixed merge_chunks with the given (start,end) list."""
    from whisperx.vads.vad import Vad

    class _ChunkedVad(Vad):
        def __init__(self):
            super().__init__(0.5)

        def preprocess_audio(self, audio):
            return audio

        @staticmethod
        def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
            return [{"start": s, "end": e, "segments": [(s, e)]} for s, e in chunks]

        def __call__(self, audio):
            return [MagicMock(start=s, end=e, speaker="UNKNOWN") for s, e in chunks]

    return _ChunkedVad()


class TestMlxWhisperPipelineTranscribe:
    def _make_pipeline(self, vad=None, language=None, suppress_tokens=None):
        if vad is None:
            vad = _stub_vad()
        return MlxWhisperPipeline(
            model_path="mlx-community/whisper-small",
            vad=vad,
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
            language=language,
            suppress_numerals=False,
            suppress_tokens=suppress_tokens,
        )

    def test_transcribe_assembles_segments_with_offset(self, monkeypatch):
        # VAD returns one chunk; mlx_whisper.transcribe returns two sub-segments.
        vad = _stub_vad()
        vad.__call__ = lambda audio: [
            MagicMock(start=1.0, end=2.0, speaker="UNKNOWN"),
        ]
        # Use a Vad subclass so preprocess_audio/merge_chunks are taken from it.
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [{"start": 1.0, "end": 2.0, "segments": [(1.0, 2.0)]}]

            def __call__(self, audio):
                return [MagicMock(start=1.0, end=2.0, speaker="UNKNOWN")]

        pipe = self._make_pipeline(vad=_Vad())
        captured: dict = {}

        def fake_transcribe(audio_slice, **kwargs):
            captured["calls"] = captured.get("calls", 0) + 1
            return {
                "language": "en",
                "segments": [
                    {"text": "hello", "start": 0.0, "end": 0.5, "avg_logprob": -0.2},
                    {"text": "world", "start": 0.5, "end": 1.0, "avg_logprob": -0.3},
                ],
            }

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake_transcribe)
        audio = np.zeros(16000 * 3, dtype=np.float32)
        result = pipe.transcribe(audio, chunk_size=30)
        assert result["language"] == "en"
        assert len(result["segments"]) == 2
        # Segment offsets are added: seg_offset (1.0) + sub_start.
        assert result["segments"][0]["start"] == 1.0
        assert result["segments"][0]["end"] == 1.5
        assert result["segments"][1]["start"] == 1.5
        assert result["segments"][0]["text"] == "hello"
        assert result["segments"][0]["avg_logprob"] == -0.2

    def test_transcribe_skips_empty_text(self, monkeypatch):
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}]

            def __call__(self, audio):
                return [MagicMock(start=0.0, end=1.0, speaker="UNKNOWN")]

        pipe = self._make_pipeline(vad=_Vad())

        def fake_transcribe(audio_slice, **kwargs):
            return {
                "language": "en",
                "segments": [
                    {"text": "  ", "start": 0.0, "end": 0.5},
                    {"text": "real", "start": 0.5, "end": 1.0},
                ],
            }

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake_transcribe)
        result = pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert len(result["segments"]) == 1
        assert result["segments"][0]["text"] == "real"

    def test_transcribe_detects_language_when_none(self, monkeypatch):
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}]

            def __call__(self, audio):
                return [MagicMock(start=0.0, end=1.0, speaker="UNKNOWN")]

        pipe = self._make_pipeline(vad=_Vad(), language=None)

        def fake_transcribe(audio_slice, **kwargs):
            return {
                "language": "fr",
                "segments": [{"text": "bonjour", "start": 0.0, "end": 1.0, "avg_logprob": -0.1}],
            }

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake_transcribe)
        result = pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert result["language"] == "fr"

    def test_transcribe_progress_callback_invoked(self, monkeypatch):
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [
                    {"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]},
                    {"start": 1.0, "end": 2.0, "segments": [(1.0, 2.0)]},
                ]

            def __call__(self, audio):
                return [MagicMock(start=0.0, end=2.0, speaker="UNKNOWN")]

        pipe = self._make_pipeline(vad=_Vad())
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda audio_slice, **k: {
                "language": "en",
                "segments": [{"text": "x", "start": 0.0, "end": 1.0}],
            },
        )
        calls = []
        pipe.transcribe(np.zeros(16000 * 3, dtype=np.float32), progress_callback=calls.append)
        assert len(calls) == 2
        assert calls[0] == 50.0
        assert calls[1] == 100.0

    def test_transcribe_passes_language_and_task(self, monkeypatch):
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}]

            def __call__(self, audio):
                return [MagicMock(start=0.0, end=1.0, speaker="UNKNOWN")]

        pipe = self._make_pipeline(vad=_Vad(), language="fr")
        captured: dict = {}

        def fake_transcribe(audio_slice, **kwargs):
            captured.update(kwargs)
            return {"language": "fr", "segments": []}

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake_transcribe)
        pipe.transcribe(np.zeros(16000, dtype=np.float32), task="translate")
        assert captured["language"] == "fr"
        assert captured["task"] == "translate"

    def test_transcribe_uses_preset_language_when_audio_is_string(self, monkeypatch, tmp_wav_path):
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [{"start": 0.0, "end": 0.01, "segments": [(0.0, 0.01)]}]

            def __call__(self, audio):
                return [MagicMock(start=0.0, end=0.01, speaker="UNKNOWN")]

        pipe = self._make_pipeline(vad=_Vad(), language="en")
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda audio_slice, **k: {"language": "en", "segments": []},
        )
        result = pipe.transcribe(tmp_wav_path)
        assert result["language"] == "en"

    def test_transcribe_verbose_prints_transcript(self, monkeypatch, capsys):
        # verbose=True prints "Transcript: [...]" for each sub-segment.
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]))

        def fake_transcribe(audio_slice, **kwargs):
            return {
                "language": "en",
                "segments": [{"text": "hello", "start": 0.0, "end": 1.0, "avg_logprob": -0.1}],
            }

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake_transcribe)
        pipe.transcribe(np.zeros(16000, dtype=np.float32), verbose=True)
        captured = capsys.readouterr()
        assert "Transcript:" in captured.out
        assert "hello" in captured.out
        assert "-->" in captured.out

    def test_transcribe_verbose_false_no_print(self, monkeypatch, capsys):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]))
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
            },
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), verbose=False)
        assert capsys.readouterr().out == ""

    def test_transcribe_print_progress_full(self, monkeypatch, capsys):
        # Two chunks, no combined_progress -> 50.00% then 100.00%.
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0), (1.0, 2.0)]))
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000 * 2, dtype=np.float32), print_progress=True)
        out = capsys.readouterr().out
        assert "Progress: 50.00%..." in out
        assert "Progress: 100.00%..." in out

    def test_transcribe_print_progress_combined_halves(self, monkeypatch, capsys):
        # combined_progress=True -> base_progress/2: 25.00% then 50.00%.
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0), (1.0, 2.0)]))
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(
            np.zeros(16000 * 2, dtype=np.float32), print_progress=True, combined_progress=True
        )
        out = capsys.readouterr().out
        assert "Progress: 25.00%..." in out
        assert "Progress: 50.00%..." in out
        assert "100.00%..." not in out

    def test_transcribe_print_progress_default_false(self, monkeypatch, capsys):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]))
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        # print_progress defaults to False.
        pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert "Progress:" not in capsys.readouterr().out

    def test_transcribe_language_falls_back_to_en(self, monkeypatch):
        # No language from args, no preset, and model returns None language.
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]), language=None)
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": None, "segments": []},
        )
        result = pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert result["language"] == "en"

    def test_transcribe_task_defaults_to_transcribe(self, monkeypatch):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]))
        captured: dict = {}
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: captured.update(k) or {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert captured["task"] == "transcribe"

    def test_transcribe_suppress_tokens_passed_through(self, monkeypatch):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]), suppress_tokens=[1, 2, 3])
        captured: dict = {}
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: captured.update(k) or {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert captured["suppress_tokens"] == [1, 2, 3]

    def test_transcribe_no_suppress_tokens_not_in_kwargs(self, monkeypatch):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]), suppress_tokens=None)
        captured: dict = {}
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: captured.update(k) or {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert "suppress_tokens" not in captured

    def test_transcribe_detected_language_logged_once(self, monkeypatch, caplog):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]), language=None)
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "de", "segments": []},
        )
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with caplog.at_level(logging.INFO, logger="whisperx"):
            pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert "Detected language: de" in caplog.text

    def test_transcribe_detected_language_not_logged_with_preset(self, monkeypatch, caplog):
        pipe = self._make_pipeline(vad=_chunked_vad([(0.0, 1.0)]), language="fr")
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "fr", "segments": []},
        )
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with caplog.at_level(logging.INFO, logger="whisperx"):
            pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert "Detected language:" not in caplog.text

    def test_non_vad_vad_model_uses_pyannote_interface(self, monkeypatch):
        # When vad_model is NOT a Vad subclass, the pyannote static interface
        # is used for preprocess_audio + merge_chunks.
        pipe = MlxWhisperPipeline(
            model_path="p",
            vad=MagicMock(),  # not a Vad
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
        )
        with (
            patch.object(asr_mod.Pyannote, "preprocess_audio", return_value=np.zeros(100)) as pre,
            patch.object(asr_mod.Pyannote, "merge_chunks") as mc,
        ):
            mc.return_value = [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}]
            pipe.vad_model = lambda audio: []
            monkeypatch.setattr(
                asr_mod.mlx_whisper, "transcribe", lambda a, **k: {"language": "en", "segments": []}
            )
            pipe.transcribe(np.zeros(16000, dtype=np.float32))
        pre.assert_called_once()
        mc.assert_called_once()


class TestLoadModel:
    def test_raises_on_invalid_vad_method(self):
        with pytest.raises(ValueError, match="Invalid vad_method"):
            load_model("small", device="cpu", vad_method="bogus")

    def test_loads_silero_vad(self, monkeypatch):
        # Silero.__init__ lazily imports detect_speech from the mlx_models
        # module; patch the source attribute so the import resolves to a mock.
        with patch("whisperx.mlx_models.silero_vad.detect_speech", lambda *a, **k: []):
            pipe = load_model(
                "small",
                device="cpu",
                vad_method="silero",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        assert isinstance(pipe, MlxWhisperPipeline)

    def test_loads_pyannote_vad(self, monkeypatch):
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model(
                "small",
                device="cpu",
                vad_method="pyannote",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        assert isinstance(pipe, MlxWhisperPipeline)

    def test_en_suffix_forces_english(self, monkeypatch):
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model(
                "small.en",
                device="cpu",
                vad_method="pyannote",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        assert pipe.preset_language == "en"

    def test_manual_vad_model_takes_priority(self):
        vad = _stub_vad()
        pipe = load_model("small", device="cpu", vad_model=vad, vad_method="silero")
        assert pipe.vad_model is vad

    def test_suppress_numerals_loads_tokenizer(self, monkeypatch):
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with patch("whisperx.asr.WhisperTokenizer") as wt_cls:
                tokenizer = MagicMock()
                tokenizer.eot = 3
                tokenizer.decode.side_effect = lambda ids: {0: " a", 1: " 5", 2: " z"}.get(
                    ids[0], "x"
                )
                wt_cls.from_pretrained.return_value = tokenizer
                pipe = load_model(
                    "small",
                    device="cpu",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                    asr_options={"suppress_numerals": True},
                )
        assert pipe.suppress_numerals is True
        assert pipe.suppress_tokens is not None
        assert 1 in pipe.suppress_tokens

    def test_cuda_device_warns(self, monkeypatch, caplog):
        # Re-enable propagation on the 'whisperx' parent so caplog (root
        # handler) captures records emitted on 'whisperx.asr'.
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("WARNING", logger="whisperx"):
                pipe = load_model(
                    "small",
                    device="cuda",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert isinstance(pipe, MlxWhisperPipeline)
        # Assert the cuda-specific warning text is present (kills == -> != mutant).
        assert "device='cuda'" in caplog.text
        assert "Apple Silicon" in caplog.text

    def test_cpu_device_does_not_warn_cuda(self, monkeypatch, caplog):
        # device != "cuda" must NOT emit the cuda warning. Kills the
        # `if device == "cuda"` -> `if device != "cuda"` mutant (which would
        # warn on cpu).
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("WARNING", logger="whisperx"):
                load_model(
                    "small",
                    device="cpu",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert "device='cuda'" not in caplog.text

    def test_compute_type_not_default_info_logs(self, monkeypatch, caplog):
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("INFO", logger="whisperx"):
                pipe = load_model(
                    "small",
                    device="cpu",
                    compute_type="float16",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert isinstance(pipe, MlxWhisperPipeline)
        # Assert the compute_type info log carries the value (kills string mutants).
        assert "float16" in caplog.text

    def test_default_compute_type_no_info_log(self, monkeypatch, caplog):
        # compute_type="default" must NOT emit the compute_type info log.
        # Kills the `if compute_type != "default"` -> `==` mutant.
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("INFO", logger="whisperx"):
                load_model(
                    "small",
                    device="cpu",
                    compute_type="default",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert "compute_type" not in caplog.text


# Default-argument and branch-coverage tests: kill default-value mutants on
# load_model and transcribe by exercising real code paths.


class TestLoadModelDefaults:
    def test_default_vad_method_is_pyannote(self):
        # Only required args; vad_method defaults to "pyannote".
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model(
                "small",
                device="cpu",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        assert isinstance(pipe, MlxWhisperPipeline)
        # Default vad_method=pyannote built a Pyannote VAD.
        from whisperx.vads.pyannote import Pyannote

        assert isinstance(pipe.vad_model, Pyannote)

    def test_default_compute_type_default_does_not_log(self, monkeypatch, caplog):
        # compute_type defaults to "default" -> the info log is skipped.
        import logging

        root_lg = logging.getLogger("whisperx")
        monkeypatch.setattr(root_lg, "propagate", True)
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("INFO", logger="whisperx"):
                pipe = load_model(
                    "small",
                    device="cpu",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert isinstance(pipe, MlxWhisperPipeline)
        assert "compute_type" not in caplog.text

    def test_default_mlx_options_drop_none_values(self):
        # mlx_options filters out None values; default asr_options produce a
        # mlx_options dict with no None entries.
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model(
                "small",
                device="cpu",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        for k, v in pipe._mlx_options.items():
            assert v is not None, f"{k} is None"

    def test_en_suffix_sets_language_en(self):
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model(
                "small.en",
                device="cpu",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        assert pipe.preset_language == "en"

    def test_default_vad_options_applied(self):
        # vad_options=None -> defaults {chunk_size:30, vad_onset:0.5, vad_offset:0.363}.
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model("small", device="cpu")
        assert pipe._vad_params["chunk_size"] == 30
        assert pipe._vad_params["vad_onset"] == 0.5
        assert pipe._vad_params["vad_offset"] == 0.363

    def test_suppress_tokens_default_neg_one(self):
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            pipe = load_model(
                "small",
                device="cpu",
                vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
            )
        assert pipe.suppress_tokens == [-1]


class TestMlxWhisperPipelineTranscribeBranches:
    def _vad(self):
        from whisperx.vads.vad import Vad

        class _Vad(Vad):
            def __init__(self):
                super().__init__(0.5)

            def preprocess_audio(self, audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}]

            def __call__(self, audio):
                from unittest.mock import MagicMock

                return [MagicMock(start=0.0, end=1.0, speaker="UNKNOWN")]

        return _Vad()

    def _pipe(self, vad=None, **kw):
        if vad is None:
            vad = self._vad()
        return MlxWhisperPipeline(
            model_path="mlx-community/whisper-small",
            vad=vad,
            vad_params={"vad_onset": 0.5, "vad_offset": 0.363},
            mlx_options={"temperature": 0.0},
            **kw,
        )

    def test_verbose_true_prints_transcript(self, monkeypatch, capsys):
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "hello", "start": 0.0, "end": 0.5, "avg_logprob": -0.1}],
            },
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), verbose=True)
        captured = capsys.readouterr()
        assert "Transcript:" in captured.out
        assert "hello" in captured.out

    def test_verbose_default_false_no_print(self, monkeypatch, capsys):
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "hello", "start": 0.0, "end": 0.5}],
            },
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32))
        captured = capsys.readouterr()
        assert "Transcript:" not in captured.out

    def test_print_progress_true_prints_progress(self, monkeypatch, capsys):
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "x", "start": 0.0, "end": 1.0}],
            },
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), print_progress=True)
        captured = capsys.readouterr()
        assert "Progress:" in captured.out
        assert "100.00%" in captured.out

    def test_print_progress_combined_offsets(self, monkeypatch, capsys):
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "x", "start": 0.0, "end": 1.0}],
            },
        )
        pipe.transcribe(
            np.zeros(16000, dtype=np.float32), print_progress=True, combined_progress=True
        )
        captured = capsys.readouterr()
        # base=100, combined -> 100/2 = 50.00%.
        assert "50.00%" in captured.out

    @staticmethod
    def _bind_log_to_stdout():
        import logging
        import sys

        root_logger = logging.getLogger("whisperx")
        root_logger.setLevel(logging.INFO)
        for h in root_logger.handlers:
            if isinstance(h, logging.StreamHandler):
                h.stream = sys.stdout

    def test_num_workers_positive_logs_value(self, monkeypatch, capsys):
        self._bind_log_to_stdout()
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), num_workers=4)
        captured = capsys.readouterr()
        # Assert the VALUE (4) is logged, killing the value->None mutant.
        assert "num_workers=4" in captured.out

    def test_num_workers_zero_no_log(self, monkeypatch, capsys):
        self._bind_log_to_stdout()
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), num_workers=0)
        captured = capsys.readouterr()
        assert "num_workers" not in captured.out

    def test_batch_size_gt_one_logs_value(self, monkeypatch, capsys):
        self._bind_log_to_stdout()
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), batch_size=8)
        captured = capsys.readouterr()
        # Assert the VALUE (8) is logged, killing the value->None mutant.
        assert "batch_size=8" in captured.out

    def test_batch_size_zero_no_log(self, monkeypatch, capsys):
        self._bind_log_to_stdout()
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), batch_size=0)
        captured = capsys.readouterr()
        assert "batch_size" not in captured.out

    def test_batch_size_one_no_log(self, monkeypatch, capsys):
        self._bind_log_to_stdout()
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {"language": "en", "segments": []},
        )
        pipe.transcribe(np.zeros(16000, dtype=np.float32), batch_size=1)
        captured = capsys.readouterr()
        assert "batch_size" not in captured.out

    def test_task_defaults_to_transcribe(self, monkeypatch):
        pipe = self._pipe()
        captured: dict = {}

        def fake(a, **k):
            captured.update(k)
            return {"language": "en", "segments": []}

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake)
        pipe.transcribe(np.zeros(16000, dtype=np.float32))
        # task=None arg -> "transcribe" passed to mlx_whisper.
        assert captured["task"] == "transcribe"

    def test_explicit_task_translate_passed_through(self, monkeypatch):
        pipe = self._pipe()
        captured: dict = {}

        def fake(a, **k):
            captured.update(k)
            return {"language": "en", "segments": []}

        monkeypatch.setattr(asr_mod.mlx_whisper, "transcribe", fake)
        pipe.transcribe(np.zeros(16000, dtype=np.float32), task="translate")
        assert captured["task"] == "translate"

    def test_language_detected_when_none(self, monkeypatch):
        pipe = self._pipe(language=None)
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "de",
                "segments": [{"text": "hallo", "start": 0.0, "end": 1.0}],
            },
        )
        result = pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert result["language"] == "de"

    def test_avg_logprop_passed_through_as_float(self, monkeypatch):
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "hi", "start": 0.0, "end": 1.0, "avg_logprob": -0.7}],
            },
        )
        result = pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert result["segments"][0]["avg_logprob"] == -0.7

    def test_avg_logprob_none_stays_none(self, monkeypatch):
        pipe = self._pipe()
        monkeypatch.setattr(
            asr_mod.mlx_whisper,
            "transcribe",
            lambda a, **k: {
                "language": "en",
                "segments": [{"text": "hi", "start": 0.0, "end": 1.0}],
            },
        )
        result = pipe.transcribe(np.zeros(16000, dtype=np.float32))
        assert result["segments"][0]["avg_logprob"] is None
