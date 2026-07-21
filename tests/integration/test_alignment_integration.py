"""Integration test: full align() flow with a mocked wav2vec2 model.

Mocks the transformers wav2vec2 model (volatile: network + torch inference)
and feeds a fixed emission matrix through the entire align() pipeline:
preprocess -> trellis -> backtrack -> merge_repeats -> word segmentation.
"""

from __future__ import annotations

import contextlib
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from whisperx.alignment import align, load_align_model
from whisperx.schema import SingleSegment


def _mock_model(emission_logits):
    """Build a mock wav2vec2 model returning fixed logits.

    emission_logits: (num_frames, vocab_size) pre-log-softmax.
    """
    model = MagicMock()
    output = MagicMock()
    output.logits = emission_logits.unsqueeze(0)
    model.return_value = output
    return model


def _emission_for(text, dictionary, num_frames, blank_id=0):
    """Build an emission where each known char peaks at evenly spaced frames."""
    vocab_size = max(dictionary.values()) + 1
    emission = torch.full((num_frames, vocab_size), -5.0)
    emission[:, blank_id] = -1.0
    chars = [(i, c) for i, c in enumerate(text) if c.lower() in dictionary]
    if not chars:
        return emission
    frames_per_char = max(1, num_frames // (len(chars) + 1))
    for seq, (_char_idx, char) in enumerate(chars):
        center = (seq + 1) * frames_per_char
        start = max(0, center - frames_per_char // 2)
        end = min(num_frames, center + frames_per_char // 2)
        token_id = dictionary[char.lower()]
        for t in range(start, end):
            emission[t, token_id] = 3.0
            emission[t, blank_id] = -3.0
    return emission


DICTIONARY = {
    "<pad>": 0,
    "a": 1,
    "b": 2,
    "c": 3,
    "d": 4,
    "e": 5,
    "f": 6,
    "g": 7,
    "h": 8,
    "i": 9,
    "k": 10,
    "l": 11,
    "m": 12,
    "n": 13,
    "o": 14,
    "p": 15,
    "r": 16,
    "s": 17,
    "t": 18,
    "u": 19,
    "w": 20,
    "x": 21,
    "|": 22,
    "y": 23,
}
METADATA = {"language": "en", "dictionary": DICTIONARY, "type": "huggingface"}


class TestAlignFullFlow:
    def test_multi_segment_alignment(self):
        """Two segments align through the full pipeline and produce words."""
        num_frames = 200
        text1 = "the cat"
        text2 = "sat down"
        e1 = _emission_for(text1, DICTIONARY, num_frames)
        e2 = _emission_for(text2, DICTIONARY, num_frames)

        model = MagicMock()
        outputs = []
        for e in (e1, e2):
            out = MagicMock()
            out.logits = e.unsqueeze(0)
            outputs.append(out)
        model.side_effect = outputs

        transcript: list[SingleSegment] = [
            {"start": 0.0, "end": 2.0, "text": text1, "avg_logprob": -0.1},
            {"start": 2.0, "end": 4.0, "text": text2, "avg_logprob": -0.2},
        ]
        audio = torch.randn(16000 * 4)

        result = align(
            transcript=transcript,
            model=model,
            align_model_metadata=METADATA,
            audio=audio,
            device="cpu",
            return_char_alignments=True,
        )
        assert "segments" in result
        assert "word_segments" in result
        assert len(result["segments"]) >= 2
        # word_segments is the concatenation across segments.
        words = [w["word"] for w in result["word_segments"]]
        assert "the" in words or "cat" in words
        # Each aligned segment carries words and chars.
        for seg in result["segments"]:
            assert "words" in seg
            assert seg["chars"] is not None or seg.get("chars") is not None
            assert isinstance(seg["start"], float)

    def test_alignment_propagates_avg_logprob(self):
        num_frames = 100
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 2.0, "text": text, "avg_logprob": -0.42}]
        audio = torch.randn(16000 * 2)
        result = align(transcript, model, METADATA, audio, "cpu")
        for seg in result["segments"]:
            assert seg.get("avg_logprob") == -0.42

    def test_segment_with_no_dictionary_chars_skipped(self):
        # A segment made entirely of unknown chars (digits) still returns
        # an aligned segment with empty words (wildcard path).
        num_frames = 50
        text = "12345"
        # No digits in dictionary; wildcard path extends emission.
        emission = torch.full((num_frames, max(DICTIONARY.values()) + 1), -5.0)
        emission[:, 0] = 0.0  # blank
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 1.0, "text": text}]
        audio = torch.randn(16000)
        result = align(transcript, model, METADATA, audio, "cpu")
        assert len(result["segments"]) == 1

    def test_segment_start_beyond_audio_duration_skipped(self):
        # A segment whose start exceeds the audio length is skipped (aligned
        # with empty words via the early-continue path).
        num_frames = 50
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        # Audio is 1s but segment starts at 5s.
        transcript = [{"start": 5.0, "end": 7.0, "text": text}]
        audio = torch.randn(16000)
        result = align(transcript, model, METADATA, audio, "cpu")
        assert len(result["segments"]) == 1
        assert result["segments"][0]["words"] == []

    def test_progress_callback_invoked(self):
        num_frames = 50
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 1.0, "text": text}]
        audio = torch.randn(16000)
        calls = []
        align(transcript, model, METADATA, audio, "cpu", progress_callback=calls.append)
        assert len(calls) == 1
        assert calls[0] == 100.0

    def test_japanese_text_joined_without_spaces(self):
        # LANGUAGES_WITHOUT_SPACES path: text not space-split, chars joined.
        ja_dict = {"<pad>": 0, "あ": 1, "い": 2, "う": 3, "え": 4, "お": 5}
        ja_meta = {"language": "ja", "dictionary": ja_dict, "type": "huggingface"}
        text = "あいう"
        num_frames = 60
        emission = torch.full((num_frames, 6), -5.0)
        emission[:, 0] = 0.0
        for i, c in enumerate(text):
            token = ja_dict[c]
            for t in range(i * 15, i * 15 + 10):
                if t < num_frames:
                    emission[t, token] = 3.0
                    emission[t, 0] = -3.0
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 1.0, "text": text}]
        audio = torch.randn(16000)
        result = align(transcript, model, ja_meta, audio, "cpu")
        assert len(result["segments"]) >= 1

    def test_numpy_audio_input_accepted(self):
        num_frames = 50
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 1.0, "text": text}]
        audio_np = np.random.randn(16000).astype(np.float32)
        result = align(transcript, model, METADATA, audio_np, "cpu")
        assert len(result["segments"]) >= 1

    def test_align_with_all_default_kwargs_mock(self):
        # Call align with only required args to kill default-value mutants.
        num_frames = 100
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 2.0, "text": text}]
        audio = torch.randn(16000 * 2)
        result = align(transcript, model, METADATA, audio, "cpu")
        assert "segments" in result
        assert "word_segments" in result
        # return_char_alignments defaults to False -> no chars attached.
        for seg in result["segments"]:
            assert seg.get("chars") is None
        # interpolate_method defaults to "nearest" -> runs without error.
        assert len(result["segments"]) >= 1

    def test_align_default_print_progress_is_false_mock(self, capsys):
        num_frames = 50
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 1.0, "text": text}]
        audio = torch.randn(16000)
        align(transcript, model, METADATA, audio, "cpu")
        captured = capsys.readouterr()
        assert "Progress:" not in captured.out

    def test_align_default_progress_callback_is_none(self):
        # progress_callback defaults to None; no callback invoked, no error.
        num_frames = 50
        text = "the cat"
        emission = _emission_for(text, DICTIONARY, num_frames)
        model = _mock_model(emission)
        transcript = [{"start": 0.0, "end": 1.0, "text": text}]
        audio = torch.randn(16000)
        # Should not raise.
        result = align(transcript, model, METADATA, audio, "cpu")
        assert len(result["segments"]) >= 1


# Real-model integration tests: load wav2vec2-base-960h from HF cache and
# exercise load_align_model + align() without mocking transformers.


REAL_ALIGN_LANG = "en"
REAL_ALIGN_REPO = "facebook/wav2vec2-base-960h"


@pytest.fixture(scope="module")
def real_align_model_and_metadata():
    """Load the real wav2vec2 align model once for the whole module."""
    model, metadata = load_align_model(REAL_ALIGN_LANG, "cpu", model_cache_only=True)
    return model, metadata


def _real_audio(seconds: float, sr: int = 16000) -> torch.Tensor:
    return torch.zeros(int(seconds * sr), dtype=torch.float32)


pytestmark = pytest.mark.slow


class TestLoadAlignModelReal:
    def test_loads_real_model_with_default_args(self):
        # Only required args; defaults for model_name/model_dir/model_cache_only
        # apply. Kills default-value mutants on those params.
        model, metadata = load_align_model(REAL_ALIGN_LANG, "cpu")
        assert metadata["language"] == REAL_ALIGN_LANG
        assert metadata["type"] == "huggingface"
        assert "|" in metadata["dictionary"]
        # model is a real torch nn.Module, not a MagicMock.
        assert hasattr(model, "forward")
        assert hasattr(model, "parameters")
        param = next(model.parameters())
        assert param.device.type == "cpu"

    def test_dictionary_is_lowercase_mapping(self):
        _model, metadata = load_align_model(REAL_ALIGN_LANG, "cpu", model_cache_only=True)
        d = metadata["dictionary"]
        assert all(k == k.lower() for k in d)

    def test_explicit_model_name_overrides_default(self):
        model, _metadata = load_align_model(
            REAL_ALIGN_LANG, "cpu", model_name=REAL_ALIGN_REPO, model_cache_only=True
        )
        assert hasattr(model, "forward")


pytestmark = pytest.mark.slow


class TestAlignRealModelDefaults:
    def test_align_with_all_default_kwargs(self, real_align_model_and_metadata):
        # ONLY required positional args; every defaulted parameter
        # (interpolate_method, return_char_alignments, print_progress,
        # combined_progress, progress_callback) takes its default value.
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
        result = align(transcript, model, metadata, _real_audio(2.0), "cpu")
        assert "segments" in result
        assert "word_segments" in result
        assert len(result["segments"]) >= 1
        # return_char_alignments defaults to False -> no chars attached.
        for seg in result["segments"]:
            assert seg.get("chars") is None
        assert len(result["word_segments"]) >= 1

    def test_align_return_char_alignments_default_is_false(self, real_align_model_and_metadata):
        # A mutant flipping the default to True would attach chars and fail.
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
        result = align(transcript, model, metadata, _real_audio(2.0), "cpu")
        for seg in result["segments"]:
            assert seg.get("chars") is None

    def test_align_return_char_alignments_true_attaches_chars(self, real_align_model_and_metadata):
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
        result = align(
            transcript, model, metadata, _real_audio(2.0), "cpu", return_char_alignments=True
        )
        assert len(result["segments"]) >= 1
        assert any(seg.get("chars") is not None for seg in result["segments"])

    def test_align_print_progress_default_is_false(self, real_align_model_and_metadata, capsys):
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        align(transcript, model, metadata, _real_audio(1.0), "cpu")
        captured = capsys.readouterr()
        assert "Progress:" not in captured.out

    def test_align_print_progress_true_emits_progress(self, real_align_model_and_metadata, capsys):
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        align(transcript, model, metadata, _real_audio(1.0), "cpu", print_progress=True)
        captured = capsys.readouterr()
        assert "Progress:" in captured.out
        assert "100.00%" in captured.out

    def test_align_combined_progress_offsets_baseline(self, real_align_model_and_metadata, capsys):
        # combined_progress=True prints (50 + base_progress / 2). With two
        # segments the first prints base=50 -> 50 + 50/2 = 75.00%. A mutant
        # that drops the +50 offset or the /2 scaling changes this value.
        model, metadata = real_align_model_and_metadata
        transcript = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        align(
            transcript,  # pyrefly: ignore[bad-argument-type]
            model,
            metadata,
            _real_audio(2.0),
            "cpu",
            print_progress=True,
            combined_progress=True,
        )
        captured = capsys.readouterr()
        assert "75.00%" in captured.out

    def test_align_progress_callback_invoked(self, real_align_model_and_metadata):
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        calls: list[float] = []
        align(transcript, model, metadata, _real_audio(1.0), "cpu", progress_callback=calls.append)
        assert calls == [100.0]

    def test_align_string_audio_path_loads_wav(
        self, real_align_model_and_metadata, tmp_wav_factory, sine_wave_audio
    ):
        # The `if isinstance(audio, str): audio = load_audio(audio)` branch.
        model, metadata = real_align_model_and_metadata
        path = tmp_wav_factory(sine_wave_audio)
        transcript = [{"start": 0.0, "end": 0.5, "text": "hello"}]
        result = align(transcript, model, metadata, path, "cpu")
        assert len(result["segments"]) >= 1

    def test_align_numpy_audio_input(self, real_align_model_and_metadata):
        # The `torch.from_numpy(audio)` branch for numpy ndarray input.
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        audio_np = np.zeros(16000, dtype=np.float32)
        result = align(transcript, model, metadata, audio_np, "cpu")
        assert len(result["segments"]) >= 1

    def test_align_2d_audio_unsqueezed(self, real_align_model_and_metadata):
        # A 2-D audio (already has a channel dim) skips unsqueeze(0).
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 1.0, "text": "hello"}]
        audio_2d = torch.zeros(1, 16000, dtype=torch.float32)
        result = align(transcript, model, metadata, audio_2d, "cpu")
        assert len(result["segments"]) >= 1

    def test_align_interpolate_method_default_is_nearest(self, real_align_model_and_metadata):
        # interpolate_method defaults to "nearest"; a mutant changing the
        # default to an invalid string would raise from pandas interpolate.
        model, metadata = real_align_model_and_metadata
        transcript = [{"start": 0.0, "end": 2.0, "text": "hello world"}]
        result = align(transcript, model, metadata, _real_audio(2.0), "cpu")
        # The default ran without error and produced aligned segments.
        assert len(result["segments"]) >= 1


pytestmark = pytest.mark.slow


class TestAlignPunktDownloadPath:
    """Exercise the NLTK punkt_tab download fallback (LookupError branch)."""

    def test_punkt_download_branch_runs_when_missing(
        self, real_align_model_and_metadata, monkeypatch
    ):
        import nltk

        model, metadata = real_align_model_and_metadata
        load_calls = {"n": 0}
        real_load = nltk.data.load

        def fake_load(path):
            load_calls["n"] += 1
            if load_calls["n"] == 1:
                raise LookupError("forced missing punkt_tab")
            return real_load(path)

        download_calls: list[tuple] = []

        def fake_download(resource, quiet=True):
            download_calls.append((resource, quiet))
            return True

        monkeypatch.setattr("whisperx.alignment.nltk_load", fake_load)
        monkeypatch.setattr(nltk, "download", fake_download)

        transcript = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
        result = align(transcript, model, metadata, _real_audio(1.0), "cpu")
        # The download was invoked with the exact resource name and quiet flag.
        assert download_calls == [("punkt_tab", True)]
        assert len(result["segments"]) >= 1

    def test_punkt_download_failure_raises_runtime_error(
        self, real_align_model_and_metadata, monkeypatch
    ):
        import nltk

        model, metadata = real_align_model_and_metadata

        def fake_load(path):
            raise LookupError("forced missing")

        def fake_download(resource, quiet=True):
            return False

        monkeypatch.setattr("whisperx.alignment.nltk_load", fake_load)
        monkeypatch.setattr(nltk, "download", fake_download)

        transcript = [{"start": 0.0, "end": 1.0, "text": "hello world"}]
        with pytest.raises(RuntimeError, match="Failed to download NLTK 'punkt_tab'"):
            align(transcript, model, metadata, _real_audio(1.0), "cpu")

    def test_punkt_non_default_language_uses_punkt_lang(
        self, real_align_model_and_metadata, monkeypatch
    ):
        # A language in PUNKT_LANGUAGES (e.g. "fr") selects the french punkt
        # model path. A mutant changing the default to "ENGLISH" requests the
        # wrong pickle.
        import nltk

        model, metadata = real_align_model_and_metadata
        fr_metadata = {
            "language": "fr",
            "dictionary": metadata["dictionary"],
            "type": "huggingface",
        }
        requested_paths: list[str] = []
        real_load = nltk.data.load

        def fake_load(path):
            requested_paths.append(path)
            return real_load(path)

        monkeypatch.setattr("whisperx.alignment.nltk_load", fake_load)
        transcript = [{"start": 0.0, "end": 1.0, "text": "bonjour"}]
        with contextlib.suppress(LookupError):
            align(transcript, model, fr_metadata, _real_audio(1.0), "cpu")
        assert any("french" in p for p in requested_paths)
