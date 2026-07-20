from typing import Optional

import mlx.core as mx

from whisperx.diarize import Segment as SegmentX
from whisperx.vads.vad import Vad
from whisperx.log_utils import get_logger

logger = get_logger(__name__)


class Silero(Vad):
    def __init__(self, **kwargs):
        logger.info("Performing voice activity detection using Silero (MLX)...")
        super().__init__(kwargs['vad_onset'])

        self.vad_onset = kwargs['vad_onset']
        self.chunk_size = kwargs['chunk_size']
        # Import here so weights load lazily on first use, not at import time.
        from whisperx.mlx_models.silero_vad import detect_speech
        self._detect_speech = detect_speech

    def __call__(self, audio, **kwargs):
        """Use Silero (MLX) to get segments of speech."""
        sample_rate = audio["sample_rate"]
        if sample_rate != 16000:
            raise ValueError("Only 16000Hz sample rate is allowed")

        waveform = audio["waveform"]
        if hasattr(waveform, "numpy"):
            waveform = waveform.numpy()
        audio_mx = mx.array(waveform, dtype=mx.float32)
        if audio_mx.ndim == 2:
            audio_mx = audio_mx[0]

        raw_segments = self._detect_speech(
            audio_mx,
            threshold=self.vad_onset,
            chunk_size=512,
            max_speech_duration_s=self.chunk_size,
        )
        return [
            SegmentX(s / sample_rate, e / sample_rate, "UNKNOWN")
            for s, e in raw_segments
        ]

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments_list, chunk_size, onset=0.5, offset: Optional[float] = None):
        assert chunk_size > 0
        if len(segments_list) == 0:
            logger.warning("No active speech found in audio")
            return []
        assert segments_list, "segments_list is empty."
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
