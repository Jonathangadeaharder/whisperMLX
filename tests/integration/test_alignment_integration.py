"""Integration test: full align() flow with a mocked wav2vec2 model.

Mocks the transformers wav2vec2 model (volatile: network + torch inference)
and feeds a fixed emission matrix through the entire align() pipeline:
preprocess -> trellis -> backtrack -> merge_repeats -> word segmentation.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import torch
from whisperx.alignment import align
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
