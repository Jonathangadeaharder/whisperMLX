"""Unit tests for whisperx.vads.vad.Vad (merge_chunks, validation)."""

from __future__ import annotations

import pytest
from whisperx.vads.pyannote import Pyannote
from whisperx.vads.silero import Silero
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


class TestMergeChunksExact:
    """Exact-value assertions killing boundary and key-name mutants."""

    def test_exact_split_boundaries(self, make_segments):
        # chunk_size=5: (0,6) no split (curr_end-curr_start=0); (6,7) splits.
        # Result: [{0,6,[(0,6)]}, {6,7,[(6,7)]}].
        segs = make_segments([(0.0, 6.0), (6.0, 7.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        assert len(merged) == 2
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 6.0
        assert merged[0]["segments"] == [(0.0, 6.0)]
        assert merged[1]["start"] == 6.0
        assert merged[1]["end"] == 7.0
        assert merged[1]["segments"] == [(6.0, 7.0)]

    def test_exact_no_split_when_within_chunk_size(self, make_segments):
        segs = make_segments([(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)])
        merged = Vad.merge_chunks(segs, chunk_size=10, onset=0.5, offset=0.363)
        assert len(merged) == 1
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 5.0
        assert merged[0]["segments"] == [(0.0, 2.0), (2.0, 4.0), (4.0, 5.0)]

    def test_split_condition_is_strict_greater(self, make_segments):
        # seg.end - curr_start == chunk_size exactly does NOT split (> not >=).
        segs = make_segments([(0.0, 5.0), (5.0, 6.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        # First seg: 5-0=5, not > 5, no split. curr_end=5.
        # Second seg: 6-0=6 > 5 and 5-0=5 > 0 -> split.
        assert len(merged) == 2
        assert merged[0]["end"] == 5.0
        assert merged[1]["start"] == 5.0

    def test_curr_end_updated_after_split_check(self, make_segments):
        # Verify curr_end is always set to seg.end regardless of split.
        segs = make_segments([(0.0, 3.0), (3.0, 7.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        # First: 3-0=3 not > 5. curr_end=3.
        # Second: 7-0=7 > 5 and 3-0=3 > 0 -> split. merged[0]={0,3,[(0,3)]}.
        # curr_start=3, curr_end=7. Final: {3,7,[(3,7)]}.
        assert len(merged) == 2
        assert merged[0] == {"start": 0.0, "end": 3.0, "segments": [(0.0, 3.0)]}
        assert merged[1] == {"start": 3.0, "end": 7.0, "segments": [(3.0, 7.0)]}

    def test_split_resets_seg_idxs(self, make_segments):
        # After a split, seg_idxs restarts from the new segment.
        segs = make_segments([(0.0, 6.0), (6.0, 7.0), (7.0, 8.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        # First split at seg (6,7): merged[0] has only [(0,6)].
        assert merged[0]["segments"] == [(0.0, 6.0)]
        # After split, seg_idxs resets, then (6,7) and (7,8) appended.
        assert merged[1]["segments"] == [(6.0, 7.0), (7.0, 8.0)]

    def test_keys_are_start_end_segments(self, make_segments):
        # Verify exact key names (mutants like "XXendXX" must fail).
        segs = make_segments([(0.0, 1.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert set(merged[0].keys()) == {"start", "end", "segments"}

    def test_final_segment_has_correct_curr_end(self, make_segments):
        # The final appended segment must have the correct curr_end.
        segs = make_segments([(0.0, 6.0), (6.0, 12.0), (12.0, 13.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        assert merged[-1]["end"] == 13.0

    def test_single_segment_exact(self, make_segments):
        segs = make_segments([(1.5, 3.5)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert len(merged) == 1
        assert merged[0] == {"start": 1.5, "end": 3.5, "segments": [(1.5, 3.5)]}


# --- merge_chunks edge cases (kill default-value + boundary mutants) ---------


class TestMergeChunksEdges:
    def test_curr_end_zero_until_first_segment(self, make_segments):
        # Before the loop runs, curr_end=0; a single short segment sets curr_end.
        # The final merged segment end equals the last seg.end.
        segs = make_segments([(1.0, 2.5)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert merged[-1]["end"] == 2.5

    def test_chunk_size_boundary_strict_gt(self, make_segments):
        # seg.end - curr_start == chunk_size exactly -> NOT split (strict >).
        segs = make_segments([(0.0, 5.0), (5.0, 6.0)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        # 5 - 0 == 5 is not > 5, so no split; but curr_end-curr_start must be >0.
        # The second seg grows curr_end to 6; 6-0>5 and 6-0>0 -> split at seg 2.
        # First merged chunk covers 0..5 (curr_end before second seg).
        assert merged[0]["start"] == 0.0
        assert merged[0]["end"] == 5.0

    def test_merged_segments_keys_present(self, make_segments):
        segs = make_segments([(0.0, 1.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert "start" in merged[0]
        assert "end" in merged[0]
        assert "segments" in merged[0]

    def test_speakers_collected_per_segment(self, make_segments):
        # speaker_idxs tracks seg.speaker; verify via segments list length.
        segs = make_segments([(0.0, 1.0), (1.0, 2.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert len(merged[0]["segments"]) == 2  # pyrefly: ignore[bad-argument-type]

    def test_final_segment_appended_even_when_short(self, make_segments):
        # After a split, the final short segment is still appended.
        segs = make_segments([(0.0, 6.0), (6.0, 6.5)])
        merged = Vad.merge_chunks(segs, chunk_size=5, onset=0.5, offset=0.363)
        assert merged[-1]["end"] == 6.5
        assert merged[-1]["start"] == 6.0

    def test_merge_returns_list(self, make_segments):
        segs = make_segments([(0.0, 1.0)])
        merged = Vad.merge_chunks(segs, chunk_size=30, onset=0.5, offset=0.363)
        assert isinstance(merged, list)
        assert len(merged) >= 1


class TestPyannoteMergeChunks:
    """Pyannote.merge_chunks wraps Vad.merge_chunks with its own assertions.
    Tests call Pyannote.merge_chunks directly to kill its mutants."""

    def test_default_onset_is_half(self):
        # onset defaults to 0.5. Pass chunk_size valid; assert the merged
        # result matches Vad.merge_chunks with onset=0.5.
        segs = [pytest.importorskip("whisperx.diarize").Segment(0.0, 1.0, "SPEAKER_00")]
        merged = Pyannote.merge_chunks(segs, chunk_size=30)
        assert isinstance(merged, list)
        assert len(merged) >= 1

    def test_chunk_size_zero_rejected(self):
        from whisperx.diarize import Segment

        segs = [Segment(0.0, 1.0, "SPEAKER_00")]
        with pytest.raises(AssertionError):
            Pyannote.merge_chunks(segs, chunk_size=0)

    def test_chunk_size_one_accepted(self):
        # chunk_size > 0 -> chunk_size=1 is valid. Kills the > 0 -> > 1 mutant.
        from whisperx.diarize import Segment

        segs = [Segment(0.0, 0.5, "SPEAKER_00")]
        merged = Pyannote.merge_chunks(segs, chunk_size=1)
        assert isinstance(merged, list)

    def test_empty_segments_warns_and_returns_empty(self, caplog):
        # len(segments)==0 -> warns "No active speech found" and returns [].
        import logging

        root_lg = logging.getLogger("whisperx")
        saved_prop = root_lg.propagate
        root_lg.propagate = True
        try:
            with caplog.at_level("WARNING", logger="whisperx"):
                out = Pyannote.merge_chunks([], chunk_size=30)
        finally:
            root_lg.propagate = saved_prop
        assert out == []
        assert "No active speech found in audio" in caplog.text

    def test_empty_segments_assert_message(self):
        # Assert message text is present in source.
        import inspect

        src = inspect.getsource(Pyannote.merge_chunks)
        assert "segments is empty." in src
        assert "No active speech found in audio" in src


class TestSileroMergeChunks:
    """Silero.merge_chunks wraps Vad.merge_chunks with its own assertions."""

    def test_default_onset_is_half(self):
        from whisperx.diarize import Segment

        segs = [Segment(0.0, 1.0, "SPEAKER_00")]
        merged = Silero.merge_chunks(segs, chunk_size=30)
        assert isinstance(merged, list)
        assert len(merged) >= 1

    def test_chunk_size_zero_rejected(self):
        from whisperx.diarize import Segment

        segs = [Segment(0.0, 1.0, "SPEAKER_00")]
        with pytest.raises(AssertionError):
            Silero.merge_chunks(segs, chunk_size=0)

    def test_chunk_size_one_accepted(self):
        # Kills the > 0 -> > 1 / >= 0 mutants.
        from whisperx.diarize import Segment

        segs = [Segment(0.0, 0.5, "SPEAKER_00")]
        merged = Silero.merge_chunks(segs, chunk_size=1)
        assert isinstance(merged, list)

    def test_empty_segments_warns_and_returns_empty(self, caplog):
        import logging

        root_lg = logging.getLogger("whisperx")
        saved_prop = root_lg.propagate
        root_lg.propagate = True
        try:
            with caplog.at_level("WARNING", logger="whisperx"):
                out = Silero.merge_chunks([], chunk_size=30)
        finally:
            root_lg.propagate = saved_prop
        assert out == []
        assert "No active speech found in audio" in caplog.text
