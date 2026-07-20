"""Unit tests for whisperx.mlx_models.pyannote_segmentation helpers.

MLX (mlx.core) is imported lazily. Pure helpers (_sigmoid, _leaky_relu,
_instance_norm, _maxpool1d, _aggregate_overlap, _lstm_layer) are tested with
small deterministic inputs. segment_audio is exercised with mocked weights and
_segmentation_forward.
"""

from __future__ import annotations

import numpy as np
from whisperx.mlx_models import pyannote_segmentation as ps


class TestSigmoid:
    def test_zero(self, mx_module):
        out = np.array(ps._sigmoid(mx_module.array([0.0])))
        assert np.isclose(out[0], 0.5)

    def test_monotonic(self, mx_module):
        xs = mx_module.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        out = np.array(ps._sigmoid(xs))
        assert np.all(np.diff(out) > 0)


class TestLeakyRelu:
    def test_positive_passthrough(self, mx_module):
        out = np.array(ps._leaky_relu(mx_module.array([2.5])))
        assert np.isclose(out[0], 2.5)

    def test_negative_slope(self, mx_module):
        out = np.array(ps._leaky_relu(mx_module.array([-2.0]), slope=0.1))
        assert np.isclose(out[0], -0.2)

    def test_zero(self, mx_module):
        out = np.array(ps._leaky_relu(mx_module.array([0.0])))
        assert np.isclose(out[0], 0.0)


class TestInstanceNorm:
    def test_normalizes_per_channel(self, mx_module):
        # (B=1, L=3, C=1): mean over L is 2, variance ~0.667.
        x = mx_module.array([[[1.0], [2.0], [3.0]]])
        w = mx_module.array([1.0])
        b = mx_module.array([0.0])
        out = np.array(ps._instance_norm(x, w, b))
        # centered around 0, std ~1
        assert out.shape == (1, 3, 1)
        assert np.isclose(out[0, 1, 0], 0.0)
        assert out[0, 0, 0] < 0
        assert out[0, 2, 0] > 0

    def test_scale_and_shift(self, mx_module):
        x = mx_module.array([[[1.0], [2.0], [3.0]]])
        w = mx_module.array([2.0])
        b = mx_module.array([1.0])
        out = np.array(ps._instance_norm(x, w, b))
        # center value (was 0) becomes 0*2+1 = 1
        assert np.isclose(out[0, 1, 0], 1.0)

    def test_multi_channel(self, mx_module):
        # (B=1, L=4, C=2)
        x = mx_module.array([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]]])
        w = mx_module.array([1.0, 1.0])
        b = mx_module.array([0.0, 0.0])
        out = np.array(ps._instance_norm(x, w, b))
        # Each channel normalized independently around its own mean.
        assert np.isclose(out[0, :, 0].mean(), 0.0, atol=1e-5)
        assert np.isclose(out[0, :, 1].mean(), 0.0, atol=1e-5)


class TestMaxpool1d:
    def test_basic_pool(self, mx_module):
        x = mx_module.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
        out = np.array(ps._maxpool1d(x, pool_size=3))
        assert out.shape == (1, 2, 1)
        assert out[0, 0, 0] == 3.0
        assert out[0, 1, 0] == 6.0

    def test_truncates_remainder(self, mx_module):
        # L=7, pool=3 -> 2 output frames (6 used, 1 dropped)
        x = mx_module.array([[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [99.0]]])
        out = np.array(ps._maxpool1d(x, pool_size=3))
        assert out.shape == (1, 2, 1)
        assert out[0, 1, 0] == 6.0

    def test_multi_channel(self, mx_module):
        x = mx_module.array([[[1.0, 9.0], [2.0, 8.0], [3.0, 7.0]]])
        out = np.array(ps._maxpool1d(x, pool_size=3))
        assert out.shape == (1, 1, 2)
        assert out[0, 0, 0] == 3.0
        assert out[0, 0, 1] == 9.0


class TestAggregateOverlap:
    def test_single_chunk_all_ones(self):
        scores = [np.ones((ps.NUM_FRAMES_PER_CHUNK, 3), dtype=np.float32)]
        num = ps.NUM_FRAMES_PER_CHUNK
        agg = ps._aggregate_overlap(scores, num)
        assert agg.shape == (num, 3)
        # With a single chunk, hamming window normalizes back to ~1.
        assert np.allclose(agg, 1.0, atol=1e-5)

    def test_two_overlapping_chunks_max(self):
        # Two chunks offset by CHUNK_STEP. Constant input -> constant output.
        scores = [np.ones((ps.NUM_FRAMES_PER_CHUNK, 3), dtype=np.float32)] * 2
        num = round(ps.CHUNK_STEP_RATIO * ps.NUM_FRAMES_PER_CHUNK) + ps.NUM_FRAMES_PER_CHUNK
        agg = ps._aggregate_overlap(scores, num)
        assert agg.shape == (num, 3)
        # In the overlap region the normalization divides by the summed hamming.
        assert np.all(np.isfinite(agg))

    def test_short_final_chunk(self):
        # Last chunk shorter than NUM_FRAMES_PER_CHUNK.
        full = np.ones((ps.NUM_FRAMES_PER_CHUNK, 3), dtype=np.float32)
        short = np.zeros((ps.NUM_FRAMES_PER_CHUNK, 3), dtype=np.float32)
        scores = [full, short]
        num = round(ps.CHUNK_STEP_RATIO * ps.NUM_FRAMES_PER_CHUNK) + ps.NUM_FRAMES_PER_CHUNK
        agg = ps._aggregate_overlap(scores, num)
        assert agg.shape[0] == num
        assert np.all(np.isfinite(agg))


class TestLstmLayer:
    def test_forward_shape(self, mx_module):
        # x: (B=1, seq=4, input_dim=4). Wx: (4*hidden, input). hidden=2.
        hidden = 2
        input_dim = 4
        seq = 4
        x = mx_module.zeros((1, seq, input_dim))
        Wx = mx_module.zeros((4 * hidden, input_dim))
        Wh = mx_module.zeros((4 * hidden, hidden))
        bias = mx_module.zeros((4 * hidden,))
        out = np.array(ps._lstm_layer(x, Wx, Wh, bias, reverse=False))
        assert out.shape == (1, seq, hidden)
        # With zero weights, all gates are sigmoid(0)=0.5 etc. Output is finite.
        assert np.all(np.isfinite(out))

    def test_reverse_produces_same_shape(self, mx_module):
        hidden = 2
        input_dim = 3
        seq = 5
        x = mx_module.ones((1, seq, input_dim)) * 0.1
        Wx = mx_module.zeros((4 * hidden, input_dim))
        Wh = mx_module.zeros((4 * hidden, hidden))
        bias = mx_module.zeros((4 * hidden,))
        fwd = np.array(ps._lstm_layer(x, Wx, Wh, bias, reverse=False))
        bwd = np.array(ps._lstm_layer(x, Wx, Wh, bias, reverse=True))
        assert fwd.shape == bwd.shape


class TestSegmentAudio:
    def test_segment_audio_with_mock_forward(self, mx_module, monkeypatch):
        # Avoid loading real weights; patch _load_weights and _segmentation_forward.
        monkeypatch.setattr(ps, "_load_weights", lambda: {"stub": True})
        monkeypatch.setattr(
            ps,
            "_segmentation_forward",
            lambda chunk, weights: mx_module.ones((1, ps.NUM_FRAMES_PER_CHUNK, 3)) * 0.9,
        )
        # 1s of audio (16000 samples) < CHUNK_SAMPLES so it gets padded.
        audio = mx_module.zeros((16000,))
        scores, frame_times = ps.segment_audio(audio)
        assert np.array(scores).shape[1] == 1
        assert np.array(scores).shape[0] > 0
        assert np.array(frame_times).shape[0] == np.array(scores).shape[0]
        # scores from constant 0.9 input aggregated back near 0.9
        assert np.all(np.abs(np.array(scores) - 0.9) < 0.2)

    def test_segment_audio_accepts_mx_array(self, mx_module, monkeypatch):
        # segment_audio expects an mx.array (the diarize pipeline converts
        # numpy upstream). Passing a raw numpy array hits an mx.pad type
        # mismatch, so we mirror real usage here.
        monkeypatch.setattr(ps, "_load_weights", dict)
        monkeypatch.setattr(
            ps,
            "_segmentation_forward",
            lambda chunk, weights: mx_module.zeros((1, ps.NUM_FRAMES_PER_CHUNK, 3)),
        )
        audio = mx_module.zeros(16000)
        scores, _frame_times = ps.segment_audio(audio)
        assert np.array(scores).shape[0] > 0
        assert np.all(np.array(scores) == 0.0)


class TestLoadWeights:
    def test_load_weights_cached(self, monkeypatch):
        class FakeSafeOpen:
            def __init__(self, path, framework):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def __iter__(self):
                return iter([])

            def get_tensor(self, k):
                return np.zeros((1,), dtype=np.float32)

        monkeypatch.setattr(ps, "safe_open", FakeSafeOpen)
        ps._load_weights.cache_clear()
        w1 = ps._load_weights()
        w2 = ps._load_weights()
        assert w1 is w2
        ps._load_weights.cache_clear()
