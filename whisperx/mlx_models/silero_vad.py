"""Silero VAD v5 forward pass in MLX.

Loads MLX-format weights from aufklarer/Silero-VAD-v5-MLX and implements the
streaming forward pass: STFT conv -> 4x Conv1d+ReLU encoder -> LSTM(128) ->
Conv1d decoder -> sigmoid. Architecture derived from the official silero-vad
tinygrad reference and the MLX weight tensor names/shapes.

MLX Conv1d convention: input (B, L, in_ch), weight (out, kernel, in),
output (B, L_out, out). Weights are stored as (out, kernel, in) so no
transpose is needed.
"""

from functools import lru_cache

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from safetensors import safe_open

_REPO = "aufklarer/Silero-VAD-v5-MLX"

# Architecture constants (from config.json).
N_FFT = 256
HOP_LENGTH = 128
CONTEXT_SIZE = 64
CUTOFF = N_FFT // 2 + 1  # 129 bins
LSTM_HIDDEN = 128


def _sigmoid(x):
    return 1.0 / (1.0 + mx.exp(-x))


@lru_cache(maxsize=1)
def _load_weights():
    path = hf_hub_download(_REPO, "model.safetensors")
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


def _lstm_cell(x, h, c, Wx, Wh, bias):
    # x: (seq, 128), h/c: (128,). Wx/Wh: (512, 128), bias: (512,).
    # PyTorch LSTM: gates = x @ Wx.T + h @ Wh.T + bias, each row is (1, 512).
    # 4 gates: i, f, g, o (PyTorch ordering).
    for t in range(x.shape[0]):
        x_t = x[t : t + 1]  # (1, 128)
        h_row = h[None, :]   # (1, 128)
        gates = x_t @ Wx.T + h_row @ Wh.T + bias
        i = _sigmoid(gates[0, 0:128])
        f = _sigmoid(gates[0, 128:256])
        g = mx.tanh(gates[0, 256:384])
        o = _sigmoid(gates[0, 384:512])
        c = f * c + i * g
        h = o * mx.tanh(c)
    return h, c


def detect_speech(audio: mx.array, threshold: float = 0.5, chunk_size: int = 512,
                  max_speech_duration_s: float = 30.0):
    """Run streaming VAD on a 1D audio array (16kHz float32).

    chunk_size is the Silero internal sample-chunk size (512).
    max_speech_duration_s caps segment length for merge_chunks compatibility.
    Returns a list of (start_sample, end_sample) speech segments.
    """
    weights = _load_weights()
    if audio.ndim == 1:
        audio = audio[None, :]

    num_samples = chunk_size
    context_size = CONTEXT_SIZE

    total = audio.shape[1]
    pad_end = (num_samples - (total % num_samples)) % num_samples
    if pad_end:
        audio = mx.pad(audio, [(0, 0), (0, pad_end)])
    audio = mx.pad(audio, [(0, 0), (context_size, 0)])

    h = mx.zeros((LSTM_HIDDEN,))
    c = mx.zeros((LSTM_HIDDEN,))
    probs = []

    for i in range(context_size, audio.shape[1], num_samples):
        chunk = audio[:, i - context_size : i + num_samples]
        out, h, c = _forward_chunk(chunk, h, c, weights)
        probs.append(float(out[0, 0]))

    return _probs_to_segments(probs, threshold, num_samples, total,
                              max_speech_duration_s, audio.shape[1])


def _forward_chunk(chunk, h, c, weights):
    # chunk: (1, 576). Pad end by CONTEXT_SIZE for STFT windowing.
    x = mx.pad(chunk, [(0, 0), (0, CONTEXT_SIZE)])
    # MLX Conv1d wants (B, L, in_ch). Input is (B, L) -> add channel dim.
    x = x[:, :, None]  # (1, L, 1)

    # STFT: conv1d with 258 output bins, kernel 256, stride 128.
    x = _conv1d(x, weights["stft.weight"], mx.zeros((258,)),
                stride=HOP_LENGTH, padding=0)
    # x: (1, L_out, 258). Split into real/imag halves for magnitude.
    real = x[:, :, :CUTOFF]
    imag = x[:, :, CUTOFF:]
    x = mx.sqrt(real * real + imag * imag)

    # Encoder: 4 conv1d + ReLU. strides [1, 2, 2, 1], kernels [3,3,3,3].
    strides = [1, 2, 2, 1]
    for i in range(4):
        x = _conv1d(x, weights[f"encoder.{i}.weight"],
                    weights[f"encoder.{i}.bias"],
                    stride=strides[i], padding=1)
        x = nn.relu(x)

    # x: (1, L_out, 128). Transpose to (L_out, 128) for LSTM.
    x = x[0]  # (L_out, 128)

    h, c = _lstm_cell(x, h, c, weights["lstm.Wx"],
                      weights["lstm.Wh"], weights["lstm.bias"])

    # Decoder: relu(h) -> conv1d(128,1,1) -> sigmoid -> mean.
    x = nn.relu(h)[None, None, :]  # (1, 1, 128)
    x = _conv1d(x, weights["decoder.weight"], weights["decoder.bias"],
                stride=1, padding=0)
    x = _sigmoid(x)
    x = x.mean(axis=1)  # (1, 1)
    return x, h, c


def _probs_to_segments(probs, threshold, num_samples, total_samples,
                       max_speech_s, padded_total):
    """Convert per-chunk probabilities to (start_sample, end_sample) segments."""
    max_samples = int(max_speech_s * 16000)
    segments = []
    in_speech = False
    start = 0
    for i, p in enumerate(probs):
        sample_start = i * num_samples
        sample_end = min((i + 1) * num_samples, total_samples)
        if p >= threshold:
            if not in_speech:
                start = sample_start
                in_speech = True
            # Cap segment length at max_speech_duration_s.
            if sample_end - start > max_samples:
                segments.append((start, sample_end))
                start = sample_end
                in_speech = True
        else:
            if in_speech:
                segments.append((start, sample_end))
                in_speech = False
    if in_speech:
        segments.append((start, total_samples))
    return segments
