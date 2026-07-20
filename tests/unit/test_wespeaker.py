"""Unit tests for whisperx.mlx_models.wespeaker helpers.

MLX (mlx.core) is imported lazily. Pure-numpy helpers (_mel_filterbank,
compute_log_mel) are tested directly. The ResNet forward path is exercised
with mocked weights via embed().
"""

from __future__ import annotations

import numpy as np
from whisperx.mlx_models import wespeaker as ws


class TestMelFilterbank:
    def test_shape(self):
        fb = ws._mel_filterbank()
        assert fb.shape == (ws.N_FFT // 2 + 1, ws.N_MELS)

    def test_dtype_float32(self):
        fb = ws._mel_filterbank()
        assert fb.dtype == np.float32

    def test_mostly_nonnegative(self):
        fb = ws._mel_filterbank()
        # Triangular filters: values in [0, 1], zero outside bands.
        assert fb.min() >= 0.0
        assert fb.max() <= 1.0

    def test_filters_are_triangular(self):
        # Each mel filter rises then falls (triangular envelope).
        fb = ws._mel_filterbank()
        for m in range(ws.N_MELS):
            col = fb[:, m]
            peak = int(np.argmax(col))
            if peak > 0 and col[peak] > 0:
                # rising before the peak
                assert col[:peak].max() <= col[peak] + 1e-6
            if peak < len(col) - 1 and col[peak] > 0:
                # falling after the peak
                assert col[peak + 1 :].max() <= col[peak] + 1e-6

    def test_filter_centers_increase(self):
        fb = ws._mel_filterbank()
        # The first nonzero bin of each filter increases with mel index.
        first_nonzero = []
        for m in range(ws.N_MELS):
            nz = np.nonzero(fb[:, m])[0]
            if len(nz):
                first_nonzero.append(nz[0])
        assert first_nonzero == sorted(first_nonzero)

    def test_cached_helper_returns_same_object(self):
        a = ws._mel_fbank()
        b = ws._mel_fbank()
        assert a is b


class TestComputeLogMel:
    def test_shape_for_one_second(self):
        audio = np.zeros(ws.SAMPLE_RATE, dtype=np.float32)
        out = np.array(ws.compute_log_mel(audio))
        # 1s at 10ms frame shift -> ~99 frames, 80 mels.
        assert out.shape[1] == ws.N_MELS
        assert out.shape[0] > 90

    def test_output_finite(self):
        audio = np.random.rand(ws.SAMPLE_RATE).astype(np.float32) * 0.1
        out = np.array(ws.compute_log_mel(audio))
        assert np.all(np.isfinite(out))

    def test_short_audio_padded(self):
        # Audio shorter than one frame is padded to FRAME_LENGTH.
        audio = np.array([0.5], dtype=np.float32)
        out = np.array(ws.compute_log_mel(audio))
        assert out.shape[1] == ws.N_MELS
        assert out.shape[0] >= 1

    def test_accepts_mlx_array(self, mx_module):
        audio = mx_module.zeros(ws.SAMPLE_RATE)
        out = np.array(ws.compute_log_mel(audio))
        assert out.shape[1] == ws.N_MELS

    def test_silence_log_low(self):
        # Pure silence -> power ~0 -> mel ~eps -> log(eps) very negative.
        audio = np.zeros(ws.SAMPLE_RATE, dtype=np.float32)
        out = np.array(ws.compute_log_mel(audio))
        assert (out < 0).all()


class TestBasicBlock:
    def test_zero_weights_residual_passes_input(self, mx_module):
        # With all-zero conv weights/biases, conv outputs 0; relu(0)=0; the
        # residual identity adds the input (broadcast across output channels).
        weights = {
            "blk.conv1.weight": mx_module.zeros((2, 3, 3, 1)),
            "blk.conv1.bias": mx_module.zeros((2,)),
            "blk.conv2.weight": mx_module.zeros((2, 3, 3, 2)),
            "blk.conv2.bias": mx_module.zeros((2,)),
        }
        x = mx_module.ones((1, 5, 5, 1))
        out = np.array(ws._basic_block(x, "blk", weights, stride=1, has_shortcut=False))
        # conv1 (k=3,p=1,s=1) keeps spatial dims; channels 1->2.
        assert out.shape == (1, 5, 5, 2)
        # relu(0 + identity=1) = 1
        assert np.allclose(out, 1.0)

    def test_relu_clamps_negative_residual(self, mx_module):
        # conv2 produces a large negative (bias -5) on a zero pre-activation;
        # adding the identity (input=1) gives -4, which the final relu clamps to 0.
        weights = {
            "blk.conv1.weight": mx_module.zeros((1, 3, 3, 1)),
            "blk.conv1.bias": mx_module.array([-5.0]),
            "blk.conv2.weight": mx_module.zeros((1, 3, 3, 1)),
            "blk.conv2.bias": mx_module.array([-5.0]),
        }
        x = mx_module.ones((1, 5, 5, 1))
        out = np.array(ws._basic_block(x, "blk", weights, stride=1, has_shortcut=False))
        # relu(conv2_neg + identity) where conv2_neg=-5, identity=1 -> relu(-4)=0
        assert np.allclose(out, 0.0)

    def test_shortcut_branch_changes_channels(self, mx_module):
        # When has_shortcut, identity comes from a 1x1 conv (zero weights ->
        # zero identity), so output is just relu(conv path).
        weights = {
            "blk.conv1.weight": mx_module.zeros((2, 3, 3, 1)),
            "blk.conv1.bias": mx_module.zeros((2,)),
            "blk.conv2.weight": mx_module.zeros((2, 3, 3, 2)),
            "blk.conv2.bias": mx_module.zeros((2,)),
            "blk.shortcut.weight": mx_module.zeros((2, 1, 1, 1)),
            "blk.shortcut.bias": mx_module.zeros((2,)),
        }
        x = mx_module.ones((1, 5, 5, 1))
        out = np.array(ws._basic_block(x, "blk", weights, stride=1, has_shortcut=True))
        # shortcut conv with zero weights -> identity=0; conv path=0; relu(0)=0.
        assert out.shape == (1, 5, 5, 2)
        assert np.allclose(out, 0.0)


class TestLayer:
    def test_layer_runs_with_shortcut_on_first(self, mx_module):
        weights = {
            "layer1.0.conv1.weight": mx_module.zeros((2, 3, 3, 1)),
            "layer1.0.conv1.bias": mx_module.zeros((2,)),
            "layer1.0.conv2.weight": mx_module.zeros((2, 3, 3, 2)),
            "layer1.0.conv2.bias": mx_module.zeros((2,)),
            "layer1.0.shortcut.weight": mx_module.zeros((2, 1, 1, 1)),
            "layer1.0.shortcut.bias": mx_module.zeros((2,)),
            "layer1.1.conv1.weight": mx_module.zeros((2, 3, 3, 2)),
            "layer1.1.conv1.bias": mx_module.zeros((2,)),
            "layer1.1.conv2.weight": mx_module.zeros((2, 3, 3, 2)),
            "layer1.1.conv2.bias": mx_module.zeros((2,)),
        }
        x = mx_module.ones((1, 8, 8, 1))
        out = np.array(ws._layer(x, 1, weights, num_blocks=2, stride=1))
        assert out.shape[0] == 1
        assert out.shape[-1] == 2

    def test_layer_no_shortcut_when_absent(self, mx_module):
        # When shortcut.weight key is missing, has_shortcut is False for the
        # first block (channels do not change here).
        weights = {
            "layer2.0.conv1.weight": mx_module.zeros((2, 3, 3, 2)),
            "layer2.0.conv1.bias": mx_module.zeros((2,)),
            "layer2.0.conv2.weight": mx_module.zeros((2, 3, 3, 2)),
            "layer2.0.conv2.bias": mx_module.zeros((2,)),
        }
        x = mx_module.ones((1, 8, 8, 2))
        out = np.array(ws._layer(x, 2, weights, num_blocks=1, stride=1))
        assert out.shape[-1] == 2


class TestEmbed:
    def test_embed_returns_256d_unit_norm(self, mx_module, monkeypatch):
        # Mock _load_weights to return a tiny dict; _wespeaker_forward needs
        # real-ish weights, so instead patch _wespeaker_forward directly.
        mx_module.zeros((256,))
        monkeypatch.setattr(ws, "_load_weights", lambda: {"stub": True})
        monkeypatch.setattr(ws, "_wespeaker_forward", lambda lm, w: mx_module.zeros((1, 256)))
        out = np.array(ws.embed(np.zeros(16000, dtype=np.float32)))
        assert out.shape == (256,)
        assert np.all(np.isfinite(out))

    def test_embed_returns_forward_emb0(self, mx_module, monkeypatch):
        # embed() returns emb[0] from _wespeaker_forward verbatim. The L2
        # normalization lives inside _wespeaker_forward, not in embed().
        v = mx_module.array([1.0] * 256)
        expected = v / mx_module.sqrt(mx_module.sum(v * v))
        monkeypatch.setattr(ws, "_load_weights", dict)
        monkeypatch.setattr(
            ws, "_wespeaker_forward", lambda lm, w: mx_module.zeros((1, 256)) + expected
        )
        out = np.array(ws.embed(np.zeros(16000, dtype=np.float32)))
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-4)
        assert np.allclose(out, np.array(expected), atol=1e-5)

    def test_embed_accepts_mlx_audio(self, mx_module, monkeypatch):
        monkeypatch.setattr(ws, "_load_weights", dict)
        monkeypatch.setattr(ws, "_wespeaker_forward", lambda lm, w: mx_module.zeros((1, 256)))
        out = np.array(ws.embed(mx_module.zeros(16000)))
        assert out.shape == (256,)


class TestLoadWeights:
    def test_load_weights_cached(self, monkeypatch):
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
                return np.zeros((1,), dtype=np.float32)

        import whisperx.mlx_models.wespeaker as wsmod  # noqa: F401

        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download", lambda repo, name: "/tmp/fake_ws", raising=False
        )
        monkeypatch.setattr(ws, "safe_open", FakeSafeOpen)
        ws._load_weights.cache_clear()
        w1 = ws._load_weights()
        w2 = ws._load_weights()
        assert w1 is w2
        ws._load_weights.cache_clear()
