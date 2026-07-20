"""Unit tests for whisperx.vads.vad.Vad (merge_chunks, validation)."""

from __future__ import annotations

import pytest
from whisperx.vads.vad import Vad


class TestVadInit:
    def test_valid_onset(self):
        Vad(vad_onset=0.5)

    def test_zero_onset_rejected(self):
        with pytest.raises(ValueError, match="decimal value"):
            Vad(vad_onset=0.0)

    def test_one_onset_rejected(self):
        with pytest.raises(ValueError, match="decimal value"):
            Vad(vad_onset=1.0)

    def test_negative_onset_rejected(self):
        with pytest.raises(ValueError, match="decimal value"):
            Vad(vad_onset=-0.1)


class TestPreprocessAudio:
    def test_preprocess_audio_is_noop(self):
        audio = object()
        assert Vad.preprocess_audio(audio) is None


class TestMergeChunks:
    def test_single_segment_kept(self, make_segments):
        segs = make_segments([(0.0, 1.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert len(merged) == 1
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 1.0
        assert merged[0]["segments"] == [(0.0, 1.0)]

    def test_multiple_short_segments_merged_into_one(self, make_segments):
        segs = make_segments([(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert len(merged) == 1
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 3.0
        # Vad.merge_chunks returns untyped dict; "segments" is a list at
        # runtime but pyrefly infers a union with the int "start"/"end" values.
        assert len(merged[0]["segments"]) == 3  # pyrefly: ignore[bad-argument-type]

    def test_split_when_chunk_size_exceeded(self, make_segments):
        # Two long segments far apart; first alone exceeds chunk_size of 5.
        segs = make_segments([(0.0, 6.0), (6.0, 7.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        # First segment (0..6) exceeds chunk_size 5 once curr_end grows.
        assert len(merged) >= 2
        assert merged[0]["start"] == 0.0

    def test_split_when_gap_exceeds_chunk(self, make_segments):
        segs = make_segments([(0.0, 2.0), (2.0, 20.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        assert len(merged) >= 2

    def test_segments_list_preserves_boundaries(self, make_segments):
        segs = make_segments([(0.0, 1.0), (1.0, 2.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert merged[0]["segments"] == [(0.0, 1.0), (1.0, 2.0)]

    def test_final_segment_appended(self, make_segments):
        # Even when splitting occurs, the final segment is always appended.
        segs = make_segments([(0.0, 6.0), (6.0, 7.0), (7.0, 8.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        last = merged[-1]
        assert last["end"] == 8.0
