"""Unit tests for whisperx.mlx_models.silero_vad helpers.

MLX (mlx.core) is imported lazily inside tests. The pure helpers (_sigmoid,
_probs_to_segments) are tested with small deterministic inputs. The forward
chunk path is exercised by mocking _load_weights and _forward_chunk so no
network/model download occurs.
"""

from __future__ import annotations

import numpy as np
from whisperx.mlx_models import silero_vad


class TestSigmoid:
    def test_zero_is_half(self, mx_module):
        out = np.array(silero_vad._sigmoid(mx_module.array([0.0])))
        assert np.isclose(out[0], 0.5, atol=1e-6)

    def test_positive_approaches_one(self, mx_module):
        out = np.array(silero_vad._sigmoid(mx_module.array([10.0])))
        assert out[0] > 0.9999

    def test_negative_approaches_zero(self, mx_module):
        out = np.array(silero_vad._sigmoid(mx_module.array([-10.0])))
        assert out[0] < 1e-4

    def test_monotonic(self, mx_module):
        xs = mx_module.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        out = np.array(silero_vad._sigmoid(xs))
        assert np.all(np.diff(out) > 0)


class TestProbsToSegments:
    def test_all_below_threshold_empty(self):
        segs = silero_vad._probs_to_segments(
            [0.1, 0.1, 0.1],
            threshold=0.5,
            num_samples=512,
            total_samples=3 * 512,
            max_speech_s=30.0,
            padded_total=3 * 512,
        )
        assert segs == []

    def test_single_speech_run(self):
        segs = silero_vad._probs_to_segments(
            [0.9, 0.9, 0.9],
            threshold=0.5,
            num_samples=512,
            total_samples=3 * 512,
            max_speech_s=30.0,
            padded_total=3 * 512,
        )
        assert len(segs) == 1
        start, end = segs[0]
        assert start == 0
        assert end == 3 * 512

    def test_speech_then_silence_then_speech(self):
        probs = [0.9, 0.1, 0.9]
        segs = silero_vad._probs_to_segments(
            probs,
            threshold=0.5,
            num_samples=512,
            total_samples=3 * 512,
            max_speech_s=30.0,
            padded_total=3 * 512,
        )
        assert len(segs) == 2
        # First speech run (chunk 0) closes at the silence chunk's sample_end
        # = (1+1)*512 = 1024.
        assert segs[0] == (0, 1024)
        # second segment starts at chunk 2 (sample 1024), ends at total
        assert segs[1][0] == 2 * 512

    def test_speech_extends_to_end(self):
        # No closing silence: segment runs to total_samples.
        segs = silero_vad._probs_to_segments(
            [0.1, 0.9, 0.9],
            threshold=0.5,
            num_samples=512,
            total_samples=3 * 512,
            max_speech_s=30.0,
            padded_total=3 * 512,
        )
        assert len(segs) == 1
        assert segs[0] == (1 * 512, 3 * 512)

    def test_max_speech_duration_caps_segment(self):
        # threshold low, max_speech 1s -> a single 3s speech run is capped.
        segs = silero_vad._probs_to_segments(
            [0.9, 0.9, 0.9],
            threshold=0.5,
            num_samples=512,
            total_samples=3 * 512,
            max_speech_s=0.03,
            padded_total=3 * 512,
        )
        # Each chunk is 512 samples = 32ms; max_speech 0.03s = 30ms = 480 samples.
        assert len(segs) >= 1
        # First segment ends near the cap.
        assert segs[0][1] <= 2 * 512

    def test_threshold_inclusive(self):
        # prob == threshold counts as speech (>=).
        segs = silero_vad._probs_to_segments(
            [0.5, 0.5],
            threshold=0.5,
            num_samples=512,
            total_samples=2 * 512,
            max_speech_s=30.0,
            padded_total=2 * 512,
        )
        assert len(segs) == 1


class TestDetectSpeech:
    def test_returns_segments_from_forward(self, mx_module, monkeypatch):
        # Mock weights so no download occurs.
        monkeypatch.setattr(silero_vad, "_load_weights", lambda: {"stub": True})
        # Forward returns (prob_array, h, c). detect_speech reads out[0,0].
        h = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        c = mx_module.zeros((silero_vad.LSTM_HIDDEN,))

        def fake_forward(chunk, h_in, c_in, weights):
            return mx_module.array([[0.9]]), h, c

        monkeypatch.setattr(silero_vad, "_forward_chunk", fake_forward)
        # 1 chunk (512 samples) of audio.
        audio = mx_module.zeros((512,))
        segs = silero_vad.detect_speech(audio, threshold=0.5, chunk_size=512)
        assert len(segs) == 1
        assert segs[0][0] == 0
        assert segs[0][1] == 512

    def test_pads_end_to_chunk_boundary(self, mx_module, monkeypatch):
        monkeypatch.setattr(silero_vad, "_load_weights", dict)
        h = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        c = mx_module.zeros((silero_vad.LSTM_HIDDEN,))

        def fake_forward(chunk, h_in, c_in, weights):
            return mx_module.array([[0.1]]), h, c

        monkeypatch.setattr(silero_vad, "_forward_chunk", fake_forward)
        # 600 samples -> padded to 1024 (2 chunks of 512).
        audio = mx_module.zeros((600,))
        segs = silero_vad.detect_speech(audio, threshold=0.5, chunk_size=512)
        assert segs == []

    def test_2d_audio_flattened(self, mx_module, monkeypatch):
        monkeypatch.setattr(silero_vad, "_load_weights", dict)
        h = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        c = mx_module.zeros((silero_vad.LSTM_HIDDEN,))

        def fake_forward(chunk, h_in, c_in, weights):
            return mx_module.array([[0.95]]), h, c

        monkeypatch.setattr(silero_vad, "_forward_chunk", fake_forward)
        audio = mx_module.zeros((1, 512))
        segs = silero_vad.detect_speech(audio, threshold=0.5, chunk_size=512)
        assert len(segs) == 1


class TestLoadWeights:
    def test_load_weights_is_cached(self, monkeypatch):
        # _load_weights is lru_cached; patch hf_hub_download to avoid network.
        import safetensors  # noqa: F401  # ensure importable

        calls = {"n": 0}

        class FakeSafeOpen:
            def __init__(self, path, framework):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def keys(self):
                return []

            def get_tensor(self, k):
                calls["n"] += 1
                return np.zeros((1,), dtype=np.float32)

        monkeypatch.setattr(silero_vad, "hf_hub_download", lambda repo, name: "/tmp/fake")
        monkeypatch.setattr(silero_vad, "safe_open", FakeSafeOpen)
        # Clear cache first so the patched path runs once.
        silero_vad._load_weights.cache_clear()
        w1 = silero_vad._load_weights()
        w2 = silero_vad._load_weights()
        assert w1 is w2
        silero_vad._load_weights.cache_clear()
