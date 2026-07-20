"""Pyannote VAD using MLX segmentation.

Replaces the pyannote.audio torch-based pipeline with an MLX forward pass
and a numpy reimplementation of the Binarize hysteresis thresholding.
"""

import os
from typing import Optional

import numpy as np

from whisperx.diarize import Segment as SegmentX
from whisperx.vads.vad import Vad
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class _SpeechSegment:
    """Minimal segment container replacing pyannote.core.Segment."""

    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end
        self.duration = end - start


class _Binarize:
    """Hysteresis thresholding with min-cut, reimplemented in numpy.

    Based on the original whisperx Binarize class (Max Bain's min-cut
    modification of pyannote's hysteresis binarization).
    """

    def __init__(self, onset=0.5, offset=None, min_duration_on=0.0,
                 min_duration_off=0.0, pad_onset=0.0, pad_offset=0.0,
                 max_duration=float("inf")):
        self.onset = onset
        self.offset = offset or onset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_duration = max_duration

    def __call__(self, scores: np.ndarray, frame_times: np.ndarray):
        """Binarize per-frame speech scores into speech segments.

        scores: (num_frames,) or (num_frames, 1) speech probabilities.
        frame_times: (num_frames,) middle timestamps per frame.
        Returns: list of _SpeechSegment.
        """
        if scores.ndim == 2:
            scores = scores[:, 0]

        timestamps = frame_times
        segments = []

        is_active = scores[0] > self.onset
        start = timestamps[0]
        curr_scores = [scores[0]]
        curr_timestamps = [start]
        t = start

        for t, y in zip(timestamps[1:], scores[1:]):
            if is_active:
                curr_duration = t - start
                if curr_duration > self.max_duration:
                    search_after = len(curr_scores) // 2
                    min_idx = search_after + int(
                        np.argmin(curr_scores[search_after:])
                    )
                    min_score_t = curr_timestamps[min_idx]
                    segments.append(_SpeechSegment(
                        start - self.pad_onset, min_score_t + self.pad_offset
                    ))
                    start = curr_timestamps[min_idx]
                    curr_scores = curr_scores[min_idx + 1:]
                    curr_timestamps = curr_timestamps[min_idx + 1:]
                elif y < self.offset:
                    segments.append(_SpeechSegment(
                        start - self.pad_onset, t + self.pad_offset
                    ))
                    start = t
                    is_active = False
                    curr_scores = []
                    curr_timestamps = []
                curr_scores.append(y)
                curr_timestamps.append(t)
            else:
                if y > self.onset:
                    start = t
                    is_active = True
                    curr_scores = [y]
                    curr_timestamps = [t]

        if is_active:
            segments.append(_SpeechSegment(
                start - self.pad_onset, t + self.pad_offset
            ))

        # Fill gaps shorter than min_duration_off and merge overlaps.
        if self.min_duration_off > 0 and len(segments) > 1:
            merged = [segments[0]]
            for seg in segments[1:]:
                prev = merged[-1]
                if seg.start - prev.end < self.min_duration_off:
                    merged[-1] = _SpeechSegment(prev.start, seg.end)
                else:
                    merged.append(seg)
            segments = merged

        # Remove segments shorter than min_duration_on.
        if self.min_duration_on > 0:
            segments = [s for s in segments if s.duration >= self.min_duration_on]

        return segments


class Pyannote(Vad):
    def __init__(self, device=None, token=None, model_fp=None, **kwargs):
        logger.info("Performing voice activity detection using Pyannote (MLX)...")
        super().__init__(kwargs["vad_onset"])
        self.vad_onset = kwargs["vad_onset"]
        self.vad_offset = kwargs.get("vad_offset", self.vad_onset)
        self.chunk_size = kwargs.get("chunk_size", 30)
        from whisperx.mlx_models.pyannote_segmentation import segment_audio
        self._segment_audio = segment_audio

    def __call__(self, audio, **kwargs):
        import mlx.core as mx

        sample_rate = audio["sample_rate"]
        waveform = audio["waveform"]
        if hasattr(waveform, "numpy"):
            waveform = waveform.numpy()
        audio_mx = mx.array(waveform, dtype=mx.float32)
        if audio_mx.ndim == 2:
            audio_mx = audio_mx[0]

        scores, frame_times = self._segment_audio(audio_mx)
        scores_np = np.array(scores)
        frame_times_np = np.array(frame_times)

        binarize = _Binarize(
            max_duration=self.chunk_size if hasattr(self, "chunk_size") else 30,
            onset=self.vad_onset,
            offset=self.vad_offset if hasattr(self, "vad_offset") else self.vad_onset,
        )
        speech_segments = binarize(scores_np, frame_times_np)
        return [
            SegmentX(float(s.start), float(s.end), "UNKNOWN")
            for s in speech_segments
        ]

    @staticmethod
    def preprocess_audio(audio):
        # MLX path takes raw numpy; no torch tensor conversion needed.
        return audio

    @staticmethod
    def merge_chunks(segments_list, chunk_size, onset=0.5, offset: Optional[float] = None):
        assert chunk_size > 0
        if len(segments_list) == 0:
            logger.warning("No active speech found in audio")
            return []
        assert segments_list, "segments_list is empty."
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
