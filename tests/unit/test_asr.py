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
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("WARNING", logger="whisperx.asr"):
                pipe = load_model(
                    "small",
                    device="cuda",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert isinstance(pipe, MlxWhisperPipeline)

    def test_compute_type_not_default_info_logs(self, monkeypatch, caplog):
        with patch(
            "whisperx.mlx_models.pyannote_segmentation.segment_audio", lambda *a, **k: ([], [])
        ):
            with caplog.at_level("INFO", logger="whisperx.asr"):
                pipe = load_model(
                    "small",
                    device="cpu",
                    compute_type="float16",
                    vad_method="pyannote",
                    vad_options={"chunk_size": 30, "vad_onset": 0.5, "vad_offset": 0.363},
                )
        assert isinstance(pipe, MlxWhisperPipeline)
