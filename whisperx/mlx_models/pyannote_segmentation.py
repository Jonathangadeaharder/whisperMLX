"""Pyannote segmentation model forward pass in MLX.

Loads weights from whisperx/assets/pyannote_segmentation_mlx.safetensors
(converted from the Pyannote PyanNet checkpoint via convert_pyannote.py)
and implements the forward pass: SincNet -> BiLSTM(4 layers) -> Linear(2)
-> classifier(3 classes) -> sigmoid.

Architecture (from pyannote.audio PyanNet):
- SincNet: InstanceNorm on waveform, then per layer:
  conv1d -> (abs if first) -> maxpool1d -> instance_norm -> leaky_relu.
  Filters: [80, 60, 60], kernels [251, 5, 5], stride [10, 1, 1],
  maxpool 3 after each. Output: (batch, frames, 60).
- BiLSTM: 4 layers, hidden 128, bidirectional. Layer 0 input 60,
  layers 1-3 input 256 (fwd+bwd concat). Output: (batch, frames, 256).
- Linear: 256 -> 128 -> 128, leaky_relu between.
- Classifier: 128 -> 3 (multi-label, sigmoid per class).

Model trained on 5s chunks at 16kHz (80000 samples), outputs 293 frames.
VAD uses 5s sliding windows with 0.5s step, overlap-add aggregation with
Hamming window, then max over the 3 speaker classes for speech probability.
"""

import os
from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open

SAMPLE_RATE = 16000
CHUNK_DURATION = 5.0
CHUNK_SAMPLES = int(CHUNK_DURATION * SAMPLE_RATE)  # 80000
CHUNK_STEP_RATIO = 0.1
CHUNK_STEP = int(CHUNK_SAMPLES * CHUNK_STEP_RATIO)  # 8000 (0.5s)
NUM_FRAMES_PER_CHUNK = 293
# Per-frame resolution: 5s / 293 frames.
FRAME_STEP = CHUNK_DURATION / NUM_FRAMES_PER_CHUNK
FRAME_DURATION = FRAME_STEP


def _sigmoid(x):
    return 1.0 / (1.0 + mx.exp(-x))


def _leaky_relu(x, slope=0.01):
    return mx.maximum(0.0, x) + slope * mx.minimum(0.0, x)


@lru_cache(maxsize=1)
def _load_weights():
    path = os.path.join(
        os.path.dirname(__file__), "..", "assets",
        "pyannote_segmentation_mlx.safetensors",
    )
    path = os.path.abspath(path)
    weights = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():
            weights[k] = mx.array(f.get_tensor(k))
    return weights


def _conv1d(x, weight, bias, stride=1, padding=0):
    # x: (B, L, in_ch). weight: (out, kernel, in). Returns (B, L_out, out).
    out_ch, kernel, in_ch = weight.shape
    conv = nn.Conv1d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel,
        stride=stride,
        padding=padding,
        bias=True,
    )
    conv.weight = weight
    conv.bias = bias
    return conv(x)


def _instance_norm(x, weight, bias, eps=1e-5):
    # InstanceNorm1d: normalize per-sample per-channel, no running stats.
    # x: (B, L, C). Normalize along L dimension for each (B, C).
    mean = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    x = (x - mean) / mx.sqrt(var + eps)
    return x * weight + bias


def _maxpool1d(x, pool_size):
    # x: (B, L, C). Max pool along L.
    B, L, C = x.shape
    new_L = L // pool_size
    x = x[:, :new_L * pool_size, :]
    x = x.reshape(B, new_L, pool_size, C)
    return x.max(axis=2)


def _lstm_layer(x, Wx, Wh, bias, reverse=False):
    # Single LSTM layer, single direction. x: (B, seq, input_dim).
    # Wx: (4*hidden, input). Wh: (4*hidden, hidden). bias: (4*hidden,).
    # Returns (B, seq, hidden).
    B, seq_len, _ = x.shape
    hidden = Wh.shape[1]
    h = mx.zeros((B, hidden))
    c = mx.zeros((B, hidden))
    outputs = []
    indices = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
    for t in indices:
        x_t = x[:, t, :]
        gates = x_t @ Wx.T + h @ Wh.T + bias
        i = _sigmoid(gates[:, 0:hidden])
        f = _sigmoid(gates[:, hidden:2*hidden])
        g = mx.tanh(gates[:, 2*hidden:3*hidden])
        o = _sigmoid(gates[:, 3*hidden:4*hidden])
        c = f * c + i * g
        h = o * mx.tanh(c)
        outputs.append(h)
    if reverse:
        outputs.reverse()
    return mx.stack(outputs, axis=1)


def _bilstm(x, fwd_weights, bwd_weights):
    """4-layer bidirectional LSTM. Each layer: fwd + bwd -> concat (256)."""
    for li in range(4):
        fwd_Wx, fwd_Wh, fwd_bias = fwd_weights[li]
        bwd_Wx, bwd_Wh, bwd_bias = bwd_weights[li]
        fwd_out = _lstm_layer(x, fwd_Wx, fwd_Wh, fwd_bias, reverse=False)
        bwd_out = _lstm_layer(x, bwd_Wx, bwd_Wh, bwd_bias, reverse=True)
        x = mx.concatenate([fwd_out, bwd_out], axis=-1)
    return x


def _segmentation_forward(audio_chunk, weights):
    """Forward pass for one 5s audio chunk.

    audio_chunk: mx.array (B, 80000) or (80000,).
    Returns: mx.array (B, 293, 3) per-class sigmoid probabilities.
    """
    if audio_chunk.ndim == 1:
        audio_chunk = audio_chunk[None, :]

    # SincNet wav_norm (InstanceNorm on raw waveform, 1 channel).
    x = audio_chunk[:, :, None]  # (B, L, 1)
    x = _instance_norm(x, weights["sincnet.wav_norm.weight"],
                       weights["sincnet.wav_norm.bias"])

    # SincNet: conv -> abs(c==0) -> pool -> norm -> leaky_relu, per layer.
    # Layer 0: 80 sinc filters, kernel 251, stride 10.
    x = _conv1d(x, weights["sincnet.conv.0.weight"],
                mx.zeros((80,)), stride=10, padding=0)
    x = mx.abs(x)
    x = _maxpool1d(x, pool_size=3)
    x = _instance_norm(x, weights["sincnet.norm.0.weight"],
                       weights["sincnet.norm.0.bias"])
    x = _leaky_relu(x)

    # Layer 1: 60 filters, kernel 5, stride 1, no padding.
    x = _conv1d(x, weights["sincnet.conv.1.weight"],
                weights["sincnet.conv.1.bias"], stride=1, padding=0)
    x = _maxpool1d(x, pool_size=3)
    x = _instance_norm(x, weights["sincnet.norm.1.weight"],
                       weights["sincnet.norm.1.bias"])
    x = _leaky_relu(x)

    # Layer 2: 60 filters, kernel 5, stride 1, no padding.
    x = _conv1d(x, weights["sincnet.conv.2.weight"],
                weights["sincnet.conv.2.bias"], stride=1, padding=0)
    x = _maxpool1d(x, pool_size=3)
    x = _instance_norm(x, weights["sincnet.norm.2.weight"],
                       weights["sincnet.norm.2.bias"])
    x = _leaky_relu(x)

    # x: (B, 293, 60). BiLSTM 4 layers, bidirectional.
    fwd_weights = [
        (weights[f"lstm_fwd.layers.{i}.Wx"],
         weights[f"lstm_fwd.layers.{i}.Wh"],
         weights[f"lstm_fwd.layers.{i}.bias"])
        for i in range(4)
    ]
    bwd_weights = [
        (weights[f"lstm_bwd.layers.{i}.Wx"],
         weights[f"lstm_bwd.layers.{i}.Wh"],
         weights[f"lstm_bwd.layers.{i}.bias"])
        for i in range(4)
    ]
    x = _bilstm(x, fwd_weights, bwd_weights)  # (B, 293, 256)

    # Linear layers: 256 -> 128 -> 128, leaky_relu between.
    x = x @ weights["linear.0.weight"].T + weights["linear.0.bias"]
    x = _leaky_relu(x)
    x = x @ weights["linear.1.weight"].T + weights["linear.1.bias"]
    x = _leaky_relu(x)

    # Classifier: 128 -> 3 classes, sigmoid (multi-label).
    x = x @ weights["classifier.weight"].T + weights["classifier.bias"]
    return _sigmoid(x)  # (B, 293, 3)


def _aggregate_overlap(chunk_scores, num_total_frames):
    """Overlap-add aggregation with Hamming window (matches pyannote).

    chunk_scores: list of (NUM_FRAMES_PER_CHUNK, 3) arrays, one per chunk.
    Returns: (num_total_frames, 3) aggregated scores.
    """
    hamming = np.hamming(NUM_FRAMES_PER_CHUNK)[:, None]
    out = np.zeros((num_total_frames, 3), dtype=np.float32)
    norm = np.zeros((num_total_frames, 1), dtype=np.float32)
    for i, scores in enumerate(chunk_scores):
        start = int(round(i * CHUNK_STEP_RATIO * NUM_FRAMES_PER_CHUNK))
        end = min(start + NUM_FRAMES_PER_CHUNK, num_total_frames)
        length = end - start
        out[start:end] += scores[:length] * hamming[:length]
        norm[start:end] += hamming[:length]
    return out / np.maximum(norm, 1e-8)


def segment_audio(audio):
    """Run pyannote segmentation on full audio.

    audio: mx.array (samples,) at 16kHz, or numpy array.
    Returns: (scores, frame_times) where scores is (num_frames, 1) speech
    probability per frame (max over speaker classes), frame_times is
    (num_frames,) middle timestamps.
    """
    weights = _load_weights()
    if not hasattr(audio, "shape"):
        audio = mx.array(audio)
    if audio.ndim == 1:
        audio = audio[None, :]

    total = audio.shape[1]
    # Pad end so the last chunk is a full 5s window.
    if total < CHUNK_SAMPLES:
        audio = mx.pad(audio, [(0, 0), (0, CHUNK_SAMPLES - total)])
        total = CHUNK_SAMPLES
    elif total % CHUNK_STEP != 0:
        pad = CHUNK_STEP - (total % CHUNK_STEP)
        if total - CHUNK_SAMPLES + pad < total:
            audio = mx.pad(audio, [(0, 0), (0, pad)])
            total = audio.shape[1]

    chunk_starts = list(range(0, total - CHUNK_SAMPLES + 1, CHUNK_STEP))
    if not chunk_starts:
        chunk_starts = [0]

    chunk_scores = []
    for start in chunk_starts:
        chunk = audio[:, start:start + CHUNK_SAMPLES]
        probs = _segmentation_forward(chunk, weights)  # (1, 293, 3)
        chunk_scores.append(np.array(probs[0]))

    # Overlap-add aggregation across 5s windows with 0.5s step.
    num_chunks = len(chunk_scores)
    num_total_frames = int(round(
        chunk_starts[-1] / SAMPLE_RATE / FRAME_STEP
    )) + NUM_FRAMES_PER_CHUNK
    agg = _aggregate_overlap(chunk_scores, num_total_frames)

    # VAD speech probability = max over speaker classes.
    scores = agg.max(axis=-1)
    frame_times = mx.array(
        [i * FRAME_STEP + FRAME_DURATION / 2 for i in range(scores.shape[0])]
    )
    return mx.array(scores)[:, None], frame_times
