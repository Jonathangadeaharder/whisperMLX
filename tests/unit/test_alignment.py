"""Unit tests for whisperx.alignment CTC helpers and load_align_model.

The pure-numpy CTC helpers (get_trellis, backtrack, merge_repeats, merge_words)
are tested with small deterministic emission matrices. load_align_model is
tested with mocked transformers Wav2Vec2 classes (volatile: network download).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from whisperx.alignment import (
    Point,
    Segment,
    backtrack,
    get_trellis,
    load_align_model,
    merge_repeats,
    merge_words,
)


class TestGetTrellis:
    def test_shape(self):
        # 3 frames, 2 tokens. emission (3, 4) with blank_id=0.
        emission = np.array(
            [[1.0, 0.1, 0.1, 0.1], [1.0, 0.1, 0.9, 0.1], [1.0, 0.1, 0.1, 0.9]],
            dtype=np.float32,
        )
        tokens = [2, 3]
        trellis = get_trellis(emission, tokens, blank_id=0)
        assert trellis.shape == (4, 3)  # (num_frame+1, num_tokens+1)

    def test_first_column_is_cumsum_of_blank(self):
        emission = np.array([[0.5, 0.2], [0.4, 0.3]], dtype=np.float32)
        trellis = get_trellis(emission, [1], blank_id=0)
        # trellis[1:, 0] is cumsum of blank scores: 0.5, then 0.9 ...
        assert np.isclose(trellis[1, 0], 0.5)
        # The last num_tokens rows of column 0 are overridden to +inf so the
        # backtrack can never exit through the bottom; for 1 token that is the
        # final row.
        assert np.isinf(trellis[-1, 0])

    def test_bottom_left_is_neg_inf(self):
        emission = np.array([[0.5, 0.2]], dtype=np.float32)
        trellis = get_trellis(emission, [1], blank_id=0)
        assert trellis[0, 1] == -float("inf")

    def test_bottom_left_inf_marker(self):
        emission = np.zeros((2, 3), dtype=np.float32)
        trellis = get_trellis(emission, [1, 2], blank_id=0)
        assert trellis[0, 1] == -float("inf")
        # last column set to inf at the bottom
        assert trellis[-2, 0] == float("inf") or trellis[-1, 0] == float("inf")

    def test_trellis_finite_outside_markers(self):
        emission = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
        trellis = get_trellis(emission, [1], blank_id=0)
        # interior values finite
        assert np.isfinite(trellis[1, 1])
        assert np.isfinite(trellis[2, 1])


class TestBacktrack:
    def test_returns_path_for_perfect_emission(self):
        # Each token peaks at its own frame.
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is not None
        assert len(path) >= 1
        # path entries are Points
        assert all(isinstance(p, Point) for p in path)
        # token indices should be 0 and 1
        assert {p.token_index for p in path} == {0, 1}

    def test_returns_none_or_path(self):
        # backtrack returns either a path or None depending on reachability;
        # we just assert the contract: None or a list of Points.
        emission = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        tokens = [1]
        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is None or (isinstance(path, list) and all(isinstance(p, Point) for p in path))

    def test_path_is_reversed_to_time_order(self):
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is not None
        times = [p.time_index for p in path]
        assert times == sorted(times)

    def test_scores_are_finite_positive(self):
        # Scores are exp(emission_value); without log-softmax normalization they
        # can exceed 1.0. The contract is that they are finite and positive.
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is not None
        for p in path:
            assert np.isfinite(p.score)
            assert p.score > 0.0


class TestMergeRepeats:
    def test_merges_consecutive_same_token(self):
        path = [
            Point(0, 0, 0.9),
            Point(0, 1, 0.8),
            Point(1, 2, 0.7),
        ]
        segments = merge_repeats(path, "ab")
        assert len(segments) == 2
        assert segments[0].label == "a"
        assert segments[0].start == 0
        assert segments[0].end == 2  # last time + 1
        assert segments[1].label == "b"
        assert segments[1].start == 2

    def test_average_score(self):
        path = [Point(0, 0, 0.8), Point(0, 1, 0.6)]
        segments = merge_repeats(path, "a")
        assert np.isclose(segments[0].score, 0.7)

    def test_single_token(self):
        path = [Point(0, 0, 1.0)]
        segments = merge_repeats(path, "a")
        assert len(segments) == 1
        assert segments[0].label == "a"

    def test_repr_contains_label(self):
        seg = Segment("a", 0, 3, 0.5)
        r = repr(seg)
        assert "a" in r
        assert "0" in r

    def test_length_property(self):
        seg = Segment("a", 2, 7, 0.5)
        assert seg.length == 5


class TestMergeWords:
    def test_splits_on_separator(self):
        # "a|b" -> two words
        path = [
            Segment("a", 0, 1, 0.9),
            Segment("|", 1, 2, 0.5),
            Segment("b", 2, 3, 0.9),
        ]
        words = merge_words(path, separator="|")
        assert len(words) == 2
        assert words[0].label == "a"
        assert words[1].label == "b"

    def test_no_separator_returns_single_word(self):
        path = [Segment("a", 0, 1, 0.9), Segment("b", 1, 2, 0.9)]
        words = merge_words(path, separator="|")
        assert len(words) == 1
        assert words[0].label == "ab"

    def test_score_is_length_weighted(self):
        path = [Segment("a", 0, 2, 1.0), Segment("b", 2, 4, 0.0)]
        words = merge_words(path, separator="|")
        # length-weighted mean of (1.0*2 + 0.0*2)/(2+2) = 0.5
        assert np.isclose(words[0].score, 0.5)


class TestLoadAlignModel:
    def _make_processor_mock(self, vocab):
        processor = MagicMock()
        tokenizer = MagicMock()
        tokenizer.get_vocab.return_value = vocab
        # processor.tokenizer is accessed at runtime.
        processor.tokenizer = tokenizer
        return processor

    def test_loads_default_model_for_english(self):
        vocab = {"<pad>": 0, "a": 1, "b": 2, "|": 3}
        processor = self._make_processor_mock(vocab)
        align_model = MagicMock()
        align_model.to.return_value = align_model

        with (
            patch("whisperx.alignment.Wav2Vec2Processor") as wp_cls,
            patch("whisperx.alignment.Wav2Vec2ForCTC") as wc_cls,
        ):
            wp_cls.from_pretrained.return_value = processor
            wc_cls.from_pretrained.return_value = align_model
            model, metadata = load_align_model("en", "cpu")
        assert model is align_model
        assert metadata["language"] == "en"
        assert metadata["dictionary"]["a"] == 1
        assert metadata["type"] == "huggingface"

    def test_explicit_model_name_overrides_default(self):
        vocab = {"<pad>": 0, "x": 5}
        processor = self._make_processor_mock(vocab)
        align_model = MagicMock()
        align_model.to.return_value = align_model

        with (
            patch("whisperx.alignment.Wav2Vec2Processor") as wp_cls,
            patch("whisperx.alignment.Wav2Vec2ForCTC") as wc_cls,
        ):
            wp_cls.from_pretrained.return_value = processor
            wc_cls.from_pretrained.return_value = align_model
            model, _metadata = load_align_model("en", "cpu", model_name="custom/model")
            assert wp_cls.from_pretrained.call_args[0][0] == "custom/model"
            assert wc_cls.from_pretrained.call_args[0][0] == "custom/model"
        assert model is align_model

    def test_unknown_language_raises(self):
        with (
            patch("whisperx.alignment.Wav2Vec2Processor"),
            patch("whisperx.alignment.Wav2Vec2ForCTC"),
            pytest.raises(ValueError, match="No default align-model"),
        ):
            load_align_model("xx", "cpu")

    def test_load_failure_wraps_value_error(self):
        with (
            patch("whisperx.alignment.Wav2Vec2Processor") as wp_cls,
            patch("whisperx.alignment.Wav2Vec2ForCTC"),
        ):
            wp_cls.from_pretrained.side_effect = Exception("boom")
            with pytest.raises(ValueError, match="could not be found"):
                load_align_model("en", "cpu")

    def test_dictionary_lowercases_chars(self):
        vocab = {"<pad>": 0, "A": 1, "|": 2}
        processor = self._make_processor_mock(vocab)
        align_model = MagicMock()
        align_model.to.return_value = align_model

        with (
            patch("whisperx.alignment.Wav2Vec2Processor") as wp_cls,
            patch("whisperx.alignment.Wav2Vec2ForCTC") as wc_cls,
        ):
            wp_cls.from_pretrained.return_value = processor
            wc_cls.from_pretrained.return_value = align_model
            _, metadata = load_align_model("en", "cpu")
        # Keys are lowercased.
        assert "a" in metadata["dictionary"]
        assert metadata["dictionary"]["a"] == 1

    def test_model_moved_to_device(self):
        vocab = {"<pad>": 0, "a": 1}
        processor = self._make_processor_mock(vocab)
        align_model = MagicMock()
        align_model.to.return_value = align_model

        with (
            patch("whisperx.alignment.Wav2Vec2Processor") as wp_cls,
            patch("whisperx.alignment.Wav2Vec2ForCTC") as wc_cls,
        ):
            wp_cls.from_pretrained.return_value = processor
            wc_cls.from_pretrained.return_value = align_model
            load_align_model("en", "mps")
        align_model.to.assert_called_once_with("mps")
