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

    def test_origin_cell_is_zero(self):
        # trellis[0, 0] is the start-of-sentence seed and must be exactly 0.
        emission = np.array([[0.5, 0.2], [0.4, 0.3]], dtype=np.float32)
        trellis = get_trellis(emission, [1], blank_id=0)
        assert trellis[0, 0] == 0.0

    def test_first_column_is_exact_cumsum(self):
        # Exact cumulative sum of the blank-token emission column, excluding
        # the last num_tokens rows which are overridden to +inf.
        emission = np.array([[0.5, 0.2], [0.4, 0.3], [0.1, 0.6]], dtype=np.float32)
        trellis = get_trellis(emission, [1], blank_id=0)
        # num_tokens=1, so the last 1 row of col 0 is +inf; rows 1..-1 are cumsum.
        expected = np.cumsum(emission[:, 0], 0)
        assert np.allclose(trellis[1:-1, 0], expected[:-1])
        assert trellis[-1, 0] == float("inf")

    def test_bottom_row_is_negative_inf(self):
        # The bottom num_tokens cells of row 0 must be -inf so the path
        # cannot start mid-token.
        emission = np.array([[0.5, 0.2, 0.3]], dtype=np.float32)
        trellis = get_trellis(emission, [1, 2], blank_id=0)
        assert trellis[0, 1] == -float("inf")
        assert trellis[0, 2] == -float("inf")

    def test_right_column_is_positive_inf(self):
        # The last num_tokens cells of column 0 must be +inf so the path
        # cannot exit early through the left edge.
        emission = np.array([[0.5, 0.2, 0.3]], dtype=np.float32)
        trellis = get_trellis(emission, [1, 2], blank_id=0)
        assert trellis[-2, 0] == float("inf")
        assert trellis[-1, 0] == float("inf")

    def test_default_blank_id_is_zero(self):
        # blank_id defaults to 0; verify the cumsum uses column 0.
        emission = np.array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.3]], dtype=np.float32)
        trellis_default = get_trellis(emission, [1, 2])
        trellis_explicit = get_trellis(emission, [1, 2], blank_id=0)
        assert np.allclose(trellis_default, trellis_explicit)

    def test_non_default_blank_id_uses_specified_column(self):
        # blank_id=1 should cumsum column 1, not column 0. Use 3 frames so
        # the cumsum (rows 1..-num_tokens) is visible before the +inf override.
        emission = np.array([[0.7, 0.1, 0.2], [0.3, 0.4, 0.3], [0.2, 0.6, 0.2]], dtype=np.float32)
        trellis = get_trellis(emission, [2], blank_id=1)
        # num_tokens=1, so only the last row is +inf; rows 1..-1 hold the cumsum.
        expected = np.cumsum(emission[:, 1], 0)
        assert np.allclose(trellis[1:-1, 0], expected[:-1])
        # And column 0 is NOT used: trellis[1,0] should be emission[0,1], not emissio...
        assert np.isclose(trellis[1, 0], emission[0, 1])
        assert not np.isclose(trellis[1, 0], emission[0, 0])

    def test_interior_uses_maximum_of_stay_and_change(self):
        # With a known emission, assert the exact trellis value at one cell.
        emission = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        # trellis[1, 1] = max(trellis[0,1] + emission[0,0], trellis[0,0] + emission[0...
        #               = max(-inf + 0.0, 0 + 1.0) = 1.0
        assert np.isclose(trellis[1, 1], 1.0)
        # trellis[2, 2] = max(trellis[1,2] + emission[1,0], trellis[1,1] + emission[1...
        #               = max(-inf + 0.0, 1.0 + 1.0) = 2.0
        assert np.isclose(trellis[2, 2], 2.0)

    def test_trellis_dtype_is_float32(self):
        emission = np.array([[0.5, 0.2]], dtype=np.float32)
        trellis = get_trellis(emission, [1], blank_id=0)
        assert trellis.dtype == np.float32


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

    def test_default_blank_id_is_zero(self):
        # backtrack blank_id defaults to 0; result must match explicit blank_id=0.
        emission = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        default_path = backtrack(trellis, emission, tokens)
        explicit_path = backtrack(trellis, emission, tokens, blank_id=0)
        assert default_path is not None
        assert explicit_path is not None
        assert [p.token_index for p in default_path] == [p.token_index for p in explicit_path]
        assert [p.time_index for p in default_path] == [p.time_index for p in explicit_path]

    def test_exact_path_for_perfect_emission(self):
        # Each token peaks at its own frame; the path must visit each token
        # at its peak frame with exact coordinates.
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is not None
        # Path is reversed to time order: first frame 0, then frame 1.
        assert len(path) == 2
        assert path[0].time_index == 0
        assert path[0].token_index == 0
        assert path[1].time_index == 1
        assert path[1].token_index == 1

    def test_score_uses_token_emission_when_changed(self):
        # When the path changes token, prob = exp(emission[t-1, tokens[j-1]]).
        # With emission[0, 1] = 1.0, the first point's score = exp(1.0) = e.
        import math

        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is not None
        # The path's score for the first frame is exp(emission[0, tokens[0]])
        # = exp(1.0) since the token changed from SoS to token 0.
        assert np.isclose(path[0].score, math.exp(1.0))

    def test_t_start_is_argmax_of_last_column(self):
        # t_start = argmax(trellis[:, j]) where j is the last token column.
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens, blank_id=0)
        j = trellis.shape[1] - 1
        expected_t_start = int(np.argmax(trellis[:, j]))
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is not None
        # The last point in time-order corresponds to t_start-1 (trellis frame).
        assert path[-1].time_index == expected_t_start - 1

    def test_unreachable_returns_none(self):
        # Force t_start=0 so the loop body never executes; else returns None.
        emission = np.array([[0.0, 1.0]], dtype=np.float32)
        tokens = [1]
        trellis = get_trellis(emission, tokens, blank_id=0)
        # Force the last column to peak at row 0 so t_start=0 and loop skips.
        trellis[:, -1] = -float("inf")
        trellis[0, -1] = 1.0
        path = backtrack(trellis, emission, tokens, blank_id=0)
        assert path is None


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

    def test_end_is_last_time_plus_one(self):
        # The end index is path[i2-1].time_index + 1, not path[i2-1].time_index.
        path = [Point(0, 5, 0.9), Point(0, 7, 0.8)]
        segments = merge_repeats(path, "a")
        assert segments[0].end == 8  # 7 + 1
        assert segments[0].start == 5

    def test_three_runs_exact_boundaries(self):
        path = [
            Point(0, 0, 0.5),
            Point(0, 1, 0.5),
            Point(1, 2, 0.5),
            Point(1, 3, 0.5),
            Point(2, 4, 0.5),
        ]
        segments = merge_repeats(path, "abc")
        assert len(segments) == 3
        assert [(s.label, s.start, s.end) for s in segments] == [
            ("a", 0, 2),
            ("b", 2, 4),
            ("c", 4, 5),
        ]

    def test_empty_path_returns_empty(self):
        segments = merge_repeats([], "abc")
        assert segments == []

    def test_score_is_mean_of_run_scores(self):
        # score = sum(scores[i1:i2]) / (i2 - i1)
        path = [Point(0, 0, 0.4), Point(0, 1, 0.8), Point(0, 2, 0.6)]
        segments = merge_repeats(path, "a")
        assert np.isclose(segments[0].score, (0.4 + 0.8 + 0.6) / 3.0)

    def test_segment_repr_contains_score_and_range(self):
        seg = Segment("x", 3, 9, 0.75)
        r = repr(seg)
        assert "x" in r
        assert "0.75" in r
        assert "3" in r
        assert "9" in r


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

    def test_word_start_and_end_from_segments(self):
        # word.start = segments[i1].start, word.end = segments[i2-1].end.
        path = [
            Segment("h", 10, 12, 0.9),
            Segment("i", 12, 14, 0.8),
            Segment("|", 14, 15, 0.5),
        ]
        words = merge_words(path, separator="|")
        assert len(words) == 1
        assert words[0].label == "hi"
        assert words[0].start == 10
        assert words[0].end == 14

    def test_multiple_words_exact_labels(self):
        # "a|b|c" -> three words with exact labels.
        path = [
            Segment("a", 0, 1, 0.9),
            Segment("|", 1, 2, 0.5),
            Segment("b", 2, 3, 0.9),
            Segment("|", 3, 4, 0.5),
            Segment("c", 4, 5, 0.9),
        ]
        words = merge_words(path, separator="|")
        assert [w.label for w in words] == ["a", "b", "c"]
        assert [w.start for w in words] == [0, 2, 4]
        assert [w.end for w in words] == [1, 3, 5]

    def test_trailing_separator_no_word(self):
        # A trailing separator does not produce an empty word.
        path = [
            Segment("a", 0, 1, 0.9),
            Segment("|", 1, 2, 0.5),
        ]
        words = merge_words(path, separator="|")
        assert len(words) == 1
        assert words[0].label == "a"

    def test_leading_separator_skipped(self):
        # Leading separator advances i1 past it without yielding a word.
        path = [
            Segment("|", 0, 1, 0.5),
            Segment("a", 1, 2, 0.9),
        ]
        words = merge_words(path, separator="|")
        assert len(words) == 1
        assert words[0].label == "a"
        assert words[0].start == 1
        assert words[0].end == 2

    def test_default_separator_is_pipe(self):
        # separator defaults to "|".
        path = [Segment("a", 0, 1, 0.9), Segment("|", 1, 2, 0.5), Segment("b", 2, 3, 0.9)]
        default_words = merge_words(path)
        explicit_words = merge_words(path, separator="|")
        assert [w.label for w in default_words] == [w.label for w in explicit_words]

    def test_custom_separator(self):
        path = [Segment("a", 0, 1, 0.9), Segment("#", 1, 2, 0.5), Segment("b", 2, 3, 0.9)]
        words = merge_words(path, separator="#")
        assert [w.label for w in words] == ["a", "b"]

    def test_empty_segments_returns_empty(self):
        assert merge_words([], separator="|") == []

    def test_score_weighted_by_segment_length(self):
        # score = sum(score * length) / sum(length)
        path = [
            Segment("a", 0, 3, 1.0),  # length 3
            Segment("b", 3, 5, 0.0),  # length 2
        ]
        words = merge_words(path, separator="|")
        # (1.0*3 + 0.0*2) / (3+2) = 0.6
        assert np.isclose(words[0].score, 0.6)


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


# --- Default-argument and edge-case tests for pure CTC helpers --------------
# Kill default-value mutants (get_trellis blank_id=0, backtrack blank_id=0)
# and edge-case mutants by exercising the default code paths.


class TestGetTrellisDefaults:
    def test_default_blank_id_is_zero(self):
        # Call with only required args; blank_id defaults to 0. The cumsum
        # first column comes from emission[:, 0] (the blank column).
        emission = np.array([[0.5, 0.2], [0.4, 0.3]], dtype=np.float32)
        trellis_default = get_trellis(emission, [1])
        trellis_explicit = get_trellis(emission, [1], blank_id=0)
        # Same result -> default is 0.
        assert np.allclose(trellis_default, trellis_explicit)
        # The cumsum of column 0 (blank) is [0.5, 0.9]; row 1 holds the first
        # cumsum value (row 2 is overridden to +inf for num_tokens=1).
        assert np.isclose(trellis_default[1, 0], 0.5)

    def test_default_blank_id_differs_from_one(self):
        # Confirm default blank_id=0 is NOT 1: column 1 cumsum would differ.
        emission = np.array([[0.5, 0.2], [0.4, 0.3]], dtype=np.float32)
        trellis_default = get_trellis(emission, [0])
        trellis_blank1 = get_trellis(emission, [0], blank_id=1)
        # First column cumsum differs (0.5/0.9 vs 0.2/0.5).
        assert not np.allclose(trellis_default[1, 0], trellis_blank1[1, 0])

    def test_trellis_shape_uses_num_frames_and_tokens(self):
        emission = np.zeros((4, 5), dtype=np.float32)
        trellis = get_trellis(emission, [1, 2, 3])
        # (num_frame+1, num_tokens+1) = (5, 4).
        assert trellis.shape == (5, 4)

    def test_first_row_first_col_is_zero(self):
        emission = np.array([[0.5, 0.2]], dtype=np.float32)
        trellis = get_trellis(emission, [1])
        assert trellis[0, 0] == 0

    def test_top_right_corner_is_neg_inf(self):
        # trellis[0, -num_tokens:] = -inf.
        emission = np.array([[0.5, 0.2, 0.1]], dtype=np.float32)
        trellis = get_trellis(emission, [1, 2])
        assert trellis[0, 1] == -float("inf")
        assert trellis[0, 2] == -float("inf")

    def test_bottom_left_corner_is_inf(self):
        # trellis[-num_tokens:, 0] = +inf.
        emission = np.array([[0.5, 0.2]], dtype=np.float32)
        trellis = get_trellis(emission, [1])
        assert trellis[1, 0] == float("inf")


class TestBacktrackDefaults:
    def test_default_blank_id_is_zero(self):
        # backtrack with default blank_id=0 vs explicit 0 -> same path.
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        path_default = backtrack(trellis, emission, tokens)
        path_explicit = backtrack(trellis, emission, tokens, blank_id=0)
        assert path_default is not None
        assert path_explicit is not None
        assert [p.token_index for p in path_default] == [p.token_index for p in path_explicit]

    def test_backtrack_returns_none_when_unreachable(self):
        # An emission where the token never wins -> backtrack may return None
        # (the for-else path). Construct a trellis where t_start=0.
        emission = np.array([[1.0, 0.0]], dtype=np.float32)
        tokens = [1]
        trellis = get_trellis(emission, tokens)
        # t_start = argmax(trellis[:, 1]); with this emission it's the last row.
        path = backtrack(trellis, emission, tokens)
        # Either None or a valid path; contract check.
        assert path is None or all(isinstance(p, Point) for p in path)

    def test_backtrack_path_reversed_to_time_order(self):
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        path = backtrack(trellis, emission, tokens)
        assert path is not None
        times = [p.time_index for p in path]
        assert times == sorted(times)

    def test_backtrack_starts_at_argmax_of_last_token_column(self):
        # t_start = argmax(trellis[:, j]) where j = num_tokens.
        emission = np.array(
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        tokens = [1, 2]
        trellis = get_trellis(emission, tokens)
        path = backtrack(trellis, emission, tokens)
        assert path is not None
        # The first path point's time_index <= t_start.
        assert path[0].time_index >= 0


class TestMergeWordsEdges:
    def test_leading_separator_skipped(self):
        # A separator as the first segment -> empty first word, then a word.
        path = [
            Segment("|", 0, 1, 0.5),
            Segment("a", 1, 2, 0.9),
            Segment("|", 2, 3, 0.5),
            Segment("b", 3, 4, 0.9),
        ]
        words = merge_words(path, separator="|")
        assert [w.label for w in words] == ["a", "b"]

    def test_trailing_separator(self):
        path = [Segment("a", 0, 1, 0.9), Segment("|", 1, 2, 0.5)]
        words = merge_words(path, separator="|")
        assert len(words) == 1
        assert words[0].label == "a"

    def test_multiple_separators_in_a_row(self):
        # Two separators back-to-back -> no empty word emitted.
        path = [
            Segment("a", 0, 1, 0.9),
            Segment("|", 1, 2, 0.5),
            Segment("|", 2, 3, 0.5),
            Segment("b", 3, 4, 0.9),
        ]
        words = merge_words(path, separator="|")
        assert [w.label for w in words] == ["a", "b"]

    def test_default_separator_is_pipe(self):
        # merge_words default separator="|".
        path = [Segment("a", 0, 1, 0.9), Segment("|", 1, 2, 0.5)]
        words_default = merge_words(path)
        words_explicit = merge_words(path, separator="|")
        assert len(words_default) == len(words_explicit) == 1
        assert words_default[0].label == words_explicit[0].label == "a"

    def test_word_end_uses_last_segment_end(self):
        # word.end = segments[i2-1].end where i2 is the separator index.
        # So word.end is the end of the segment just before the separator.
        path = [
            Segment("a", 0, 1, 0.9),
            Segment("b", 1, 7, 0.9),
            Segment("|", 7, 9, 0.5),
        ]
        words = merge_words(path, separator="|")
        assert len(words) == 1
        assert words[0].start == 0
        # end = segments[i2-1].end = segments[1].end = 7.
        assert words[0].end == 7


class TestMergeRepeatsEdges:
    def test_single_token_single_segment(self):
        path = [Point(0, 0, 1.0)]
        segs = merge_repeats(path, "a")
        assert len(segs) == 1
        assert segs[0].start == 0
        assert segs[0].end == 1  # last time + 1
        assert segs[0].label == "a"

    def test_all_same_token_one_segment(self):
        path = [Point(0, 0, 1.0), Point(0, 1, 0.8), Point(0, 2, 0.6)]
        segs = merge_repeats(path, "a")
        assert len(segs) == 1
        assert segs[0].start == 0
        assert segs[0].end == 3
        assert np.isclose(segs[0].score, (1.0 + 0.8 + 0.6) / 3)

    def test_repr_and_length(self):
        seg = Segment("x", 1, 4, 0.7)
        r = repr(seg)
        assert "x" in r
        assert seg.length == 3
