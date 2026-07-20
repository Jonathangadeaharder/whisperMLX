"""Unit tests for the full MLX forward paths (silero, pyannote, wespeaker).

These exercise the complete forward pass with small random but correctly-shaped
weights, verifying the architecture wiring (shapes, no NaNs, finite output).
Weights are generated, not loaded, so no network/runtime model download occurs.
"""

from __future__ import annotations

import numpy as np
from whisperx.mlx_models import pyannote_segmentation as ps
from whisperx.mlx_models import silero_vad
from whisperx.mlx_models import wespeaker as ws


def _silero_weights(mx):
    mx.random.seed(0)
    return {
        "stft.weight": mx.random.normal((258, silero_vad.N_FFT, 1)) * 0.01,
        "encoder.0.weight": mx.random.normal((128, 3, silero_vad.CUTOFF)) * 0.01,
        "encoder.0.bias": mx.zeros((128,)),
        "encoder.1.weight": mx.random.normal((128, 3, 128)) * 0.01,
        "encoder.1.bias": mx.zeros((128,)),
        "encoder.2.weight": mx.random.normal((128, 3, 128)) * 0.01,
        "encoder.2.bias": mx.zeros((128,)),
        "encoder.3.weight": mx.random.normal((128, 3, 128)) * 0.01,
        "encoder.3.bias": mx.zeros((128,)),
        "lstm.Wx": mx.random.normal((512, silero_vad.LSTM_HIDDEN)) * 0.01,
        "lstm.Wh": mx.random.normal((512, silero_vad.LSTM_HIDDEN)) * 0.01,
        "lstm.bias": mx.zeros((512,)),
        "decoder.weight": mx.random.normal((1, 1, 128)) * 0.01,
        "decoder.bias": mx.zeros((1,)),
    }


class TestSileroForwardChunk:
    def test_forward_chunk_returns_scalar_prob(self, mx_module):
        weights = _silero_weights(mx_module)
        h = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        c = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        chunk = mx_module.zeros((1, 576))
        out, _h2, _c2 = silero_vad._forward_chunk(chunk, h, c, weights)
        arr = np.array(out)
        assert arr.shape == (1, 1)
        assert np.isfinite(arr).all()
        # sigmoid output in [0, 1]
        assert 0.0 <= arr[0, 0] <= 1.0

    def test_forward_chunk_updates_hidden_state(self, mx_module):
        weights = _silero_weights(mx_module)
        h = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        c = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        chunk = mx_module.ones((1, 576)) * 0.1
        _, h2, c2 = silero_vad._forward_chunk(chunk, h, c, weights)
        # Hidden state should change from zero (input is nonzero).
        assert not np.allclose(np.array(h2), 0.0)
        assert not np.allclose(np.array(c2), 0.0)

    def test_forward_chunk_handles_nonzero_input(self, mx_module):
        # With nonzero input the forward pass still produces a finite prob.
        weights = _silero_weights(mx_module)
        h = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        c = mx_module.zeros((silero_vad.LSTM_HIDDEN,))
        chunk_signal = mx_module.ones((1, 576)) * 0.5
        out, _h2, _c2 = silero_vad._forward_chunk(chunk_signal, h, c, weights)
        arr = np.array(out)
        assert np.isfinite(arr).all()
        assert 0.0 <= arr[0, 0] <= 1.0

    def test_conv1d_helper_shape(self, mx_module):
        # _conv1d with a 1x1 identity-like kernel preserves input.
        weight = mx_module.zeros((2, 1, 2))
        bias = mx_module.zeros((2,))
        x = mx_module.ones((1, 5, 2))
        out = np.array(silero_vad._conv1d(x, weight, bias, stride=1, padding=0))
        assert out.shape == (1, 5, 2)

    def test_lstm_cell_carries_state(self, mx_module):
        weights = _silero_weights(mx_module)
        Wx = weights["lstm.Wx"]
        Wh = weights["lstm.Wh"]
        bias = weights["lstm.bias"]
        x = mx_module.ones((3, 128)) * 0.1
        h = mx_module.zeros((128,))
        c = mx_module.zeros((128,))
        h2, c2 = silero_vad._lstm_cell(x, h, c, Wx, Wh, bias)
        assert np.array(h2).shape == (128,)
        assert np.array(c2).shape == (128,)
        assert np.all(np.isfinite(np.array(h2)))


class TestSileroDetectSpeechReal:
    def test_detect_speech_with_real_forward(self, mx_module, monkeypatch):
        # Patch only _load_weights; let _forward_chunk run for real.
        weights = _silero_weights(mx_module)
        monkeypatch.setattr(silero_vad, "_load_weights", lambda: weights)
        # 1 chunk of zeros -> probability ~0.5 -> above 0.5 threshold -> speech.
        audio = mx_module.zeros((512,))
        segs = silero_vad.detect_speech(audio, threshold=0.5, chunk_size=512)
        assert isinstance(segs, list)
        # With threshold 0.5 and sigmoid(0)=0.5, >= holds -> one segment.
        assert len(segs) == 1

    def test_detect_speech_high_threshold_no_speech(self, mx_module, monkeypatch):
        weights = _silero_weights(mx_module)
        monkeypatch.setattr(silero_vad, "_load_weights", lambda: weights)
        audio = mx_module.zeros((512,))
        segs = silero_vad.detect_speech(audio, threshold=0.99, chunk_size=512)
        # sigmoid(0) = 0.5 < 0.99 -> no speech.
        assert segs == []


def _pyannote_weights(mx):
    """Build correctly-shaped random weights for the segmentation forward pass."""
    mx.random.seed(0)

    def w(*shape):
        return mx.random.normal(shape) * 0.01

    weights = {
        "sincnet.wav_norm.weight": mx.ones((1,)),
        "sincnet.wav_norm.bias": mx.zeros((1,)),
        "sincnet.conv.0.weight": w(80, 251, 1),
        "sincnet.norm.0.weight": mx.ones((80,)),
        "sincnet.norm.0.bias": mx.zeros((80,)),
        "sincnet.conv.1.weight": w(60, 5, 80),
        "sincnet.conv.1.bias": mx.zeros((60,)),
        "sincnet.norm.1.weight": mx.ones((60,)),
        "sincnet.norm.1.bias": mx.zeros((60,)),
        "sincnet.conv.2.weight": w(60, 5, 60),
        "sincnet.conv.2.bias": mx.zeros((60,)),
        "sincnet.norm.2.weight": mx.ones((60,)),
        "sincnet.norm.2.bias": mx.zeros((60,)),
        "linear.0.weight": w(128, 256),
        "linear.0.bias": mx.zeros((128,)),
        "linear.1.weight": w(128, 128),
        "linear.1.bias": mx.zeros((128,)),
        "classifier.weight": w(3, 128),
        "classifier.bias": mx.zeros((3,)),
    }
    for i in range(4):
        # Layer 0 input=60, layers 1-3 input=256 (fwd+bwd concat from prior layer).
        in_dim = 60 if i == 0 else 256
        weights[f"lstm_fwd.layers.{i}.Wx"] = w(512, in_dim)
        weights[f"lstm_fwd.layers.{i}.Wh"] = w(512, 128)
        weights[f"lstm_fwd.layers.{i}.bias"] = mx.zeros((512,))
        weights[f"lstm_bwd.layers.{i}.Wx"] = w(512, in_dim)
        weights[f"lstm_bwd.layers.{i}.Wh"] = w(512, 128)
        weights[f"lstm_bwd.layers.{i}.bias"] = mx.zeros((512,))
    return weights


class TestPyannoteSegmentationForward:
    def test_segmentation_forward_shape(self, mx_module):
        weights = _pyannote_weights(mx_module)
        # 5s chunk at 16kHz = 80000 samples.
        chunk = mx_module.zeros((1, ps.CHUNK_SAMPLES))
        out = np.array(ps._segmentation_forward(chunk, weights))
        assert out.shape == (1, ps.NUM_FRAMES_PER_CHUNK, 3)
        assert np.all(np.isfinite(out))
        # sigmoid output in [0, 1]
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_segmentation_forward_1d_input(self, mx_module):
        weights = _pyannote_weights(mx_module)
        chunk = mx_module.zeros((ps.CHUNK_SAMPLES,))
        out = np.array(ps._segmentation_forward(chunk, weights))
        assert out.shape == (1, ps.NUM_FRAMES_PER_CHUNK, 3)

    def test_bilstm_shape(self, mx_module):
        weights = _pyannote_weights(mx_module)
        fwd = [
            (
                weights[f"lstm_fwd.layers.{i}.Wx"],
                weights[f"lstm_fwd.layers.{i}.Wh"],
                weights[f"lstm_fwd.layers.{i}.bias"],
            )
            for i in range(4)
        ]
        bwd = [
            (
                weights[f"lstm_bwd.layers.{i}.Wx"],
                weights[f"lstm_bwd.layers.{i}.Wh"],
                weights[f"lstm_bwd.layers.{i}.bias"],
            )
            for i in range(4)
        ]
        x = mx_module.zeros((1, 10, 60))
        out = np.array(ps._bilstm(x, fwd, bwd))
        assert out.shape == (1, 10, 256)


class TestPyannoteSegmentAudioReal:
    def test_segment_audio_full_path(self, mx_module, monkeypatch):
        weights = _pyannote_weights(mx_module)
        monkeypatch.setattr(ps, "_load_weights", lambda: weights)
        # 1s of audio padded to 5s.
        audio = mx_module.zeros((16000,))
        scores, frame_times = ps.segment_audio(audio)
        scores_np = np.array(scores)
        assert scores_np.ndim == 2
        assert scores_np.shape[1] == 1
        assert np.all(np.isfinite(scores_np))
        assert scores_np.shape[0] == np.array(frame_times).shape[0]


def _wespeaker_weights(mx):
    mx.random.seed(0)

    def w(*shape):
        return mx.random.normal(shape) * 0.01

    weights = {
        "conv1.weight": w(32, 3, 3, 1),
        "conv1.bias": mx.zeros((32,)),
        "embedding.weight": w(256, 5120),
        "embedding.bias": mx.zeros((256,)),
    }

    def _layer_weights(layer_idx, num_blocks, in_ch, out_ch, first_stride):
        for b in range(num_blocks):
            prefix = f"layer{layer_idx}.{b}"
            stride = first_stride if b == 0 else 1
            weights[f"{prefix}.conv1.weight"] = w(out_ch, 3, 3, in_ch)
            weights[f"{prefix}.conv1.bias"] = mx.zeros((out_ch,))
            weights[f"{prefix}.conv2.weight"] = w(out_ch, 3, 3, out_ch)
            weights[f"{prefix}.conv2.bias"] = mx.zeros((out_ch,))
            if b == 0 and stride != 1:
                weights[f"{prefix}.shortcut.weight"] = w(out_ch, 1, 1, in_ch)
                weights[f"{prefix}.shortcut.bias"] = mx.zeros((out_ch,))
            in_ch = out_ch

    _layer_weights(1, 3, 32, 32, 1)
    _layer_weights(2, 4, 32, 64, 2)
    _layer_weights(3, 6, 64, 128, 2)
    _layer_weights(4, 3, 128, 256, 2)
    return weights


class TestWespeakerForward:
    def test_wespeaker_forward_shape(self, mx_module):
        weights = _wespeaker_weights(mx_module)
        # log_mel (T, 80) -> (1, T, 80, 1)
        log_mel = mx_module.zeros((50, ws.N_MELS))
        out = np.array(ws._wespeaker_forward(log_mel, weights))
        assert out.shape == (1, 256)
        assert np.all(np.isfinite(out))
        # L2-normalized
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-4)

    def test_wespeaker_forward_3d_input(self, mx_module):
        weights = _wespeaker_weights(mx_module)
        log_mel = mx_module.zeros((50, ws.N_MELS, 1))
        out = np.array(ws._wespeaker_forward(log_mel, weights))
        assert out.shape == (1, 256)

    def test_embed_full_path(self, mx_module, monkeypatch):
        weights = _wespeaker_weights(mx_module)
        monkeypatch.setattr(ws, "_load_weights", lambda: weights)
        out = np.array(ws.embed(np.zeros(ws.SAMPLE_RATE, dtype=np.float32)))
        assert out.shape == (256,)
        assert np.isclose(np.linalg.norm(out), 1.0, atol=1e-4)

    def test_conv2d_helper_shape(self, mx_module):
        weight = mx_module.zeros((4, 3, 3, 2))
        bias = mx_module.zeros((4,))
        x = mx_module.ones((1, 5, 5, 2))
        out = np.array(ws._conv2d(x, weight, bias, stride=1, padding=1))
        assert out.shape == (1, 5, 5, 4)
