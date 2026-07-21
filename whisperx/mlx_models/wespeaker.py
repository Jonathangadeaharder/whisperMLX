"""WeSpeaker ResNet34-LM speaker embedding model forward pass in MLX.

Loads weights from aufklarer/WeSpeaker-ResNet34-LM-MLX (safetensors)
and computes 256-dim L2-normalized speaker embeddings from log-mel
spectrograms.

Architecture (from config.json + model card):
- conv1: Conv2d(1->32, k=3, p=1) + ReLU
- layer1: 3 BasicBlocks (32->32)
- layer2: 4 BasicBlocks (32->64, first block stride 2)
- layer3: 6 BasicBlocks (64->128, first block stride 2)
- layer4: 3 BasicBlocks (128->256, first block stride 2)
- Statistics pooling: mean + std concat -> [B, 5120]
- embedding: Linear(5120->256) + L2 normalize

Each BasicBlock: conv1(k=3,p=1) -> ReLU -> conv2(k=3,p=1) + shortcut
(shortcut is 1x1 conv when channels change, else identity) -> ReLU.
BatchNorm was fused into Conv2d at conversion time (bias only).

Input: log-mel spectrogram [B, T, 80, 1] at 16kHz.
Mel params: 80 fbank, 25ms frame, 10ms shift, hamming window.
Output: [B, 256] L2-normalized embeddings.
"""

from functools import lru_cache

import mlx.core as mx  # pyrefly: ignore[missing-import]
import numpy as np
from mlx import nn
from safetensors import safe_open

SAMPLE_RATE = 16000
N_MELS = 80
FRAME_LENGTH = int(0.025 * SAMPLE_RATE)  # 400 samples
FRAME_SHIFT = int(0.010 * SAMPLE_RATE)  # 160 samples
N_FFT = 512


def _mel_filterbank(
    n_mels=N_MELS,
    n_fft=N_FFT,  # noqa: ARG001
    sample_rate=SAMPLE_RATE,
    f_min=0.0,
    f_max=None,
):
    """Compute mel filterbank matrix (n_fft//2+1, n_mels)."""
    if f_max is None:
        f_max = sample_rate / 2

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(f_min)
    mel_max = hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((N_FFT + 1) * hz_points / sample_rate).astype(int)

    n_freqs = N_FFT // 2 + 1
    fbank = np.zeros((n_freqs, n_mels), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left = bin_points[m - 1]
        center = bin_points[m]
        right = bin_points[m + 1]
        for k in range(left, center):
            if center > left:
                fbank[k, m - 1] = (k - left) / (center - left)
        for k in range(center, right):
            if right > center:
                fbank[k, m - 1] = (right - k) / (right - center)
    return fbank


@lru_cache(maxsize=1)
def _mel_fbank():
    return _mel_filterbank()


def compute_log_mel(audio, sample_rate=SAMPLE_RATE):  # noqa: ARG001
    """Compute log-mel spectrogram from raw audio.

    audio: mx.array (samples,) or numpy array, 16kHz.
    Returns: mx.array (n_frames, N_MELS) log-mel features.
    """
    if hasattr(audio, "shape") and isinstance(audio, mx.array):
        audio = np.array(audio)
    audio = np.asarray(audio, dtype=np.float32)

    # Pre-emphasis (standard WeSpeaker preprocessing).
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Pad so framing fits exactly, avoiding out-of-bounds reads.
    n_samples = len(emphasized)
    if n_samples < FRAME_LENGTH:
        emphasized = np.pad(emphasized, (0, FRAME_LENGTH - n_samples))
        n_samples = FRAME_LENGTH
    remainder = (n_samples - FRAME_LENGTH) % FRAME_SHIFT
    if remainder > 0:
        emphasized = np.pad(emphasized, (0, FRAME_SHIFT - remainder))
        n_samples = len(emphasized)
    n_frames = 1 + (n_samples - FRAME_LENGTH) // FRAME_SHIFT
    frames = np.empty((n_frames, FRAME_LENGTH), dtype=np.float32)
    for i in range(n_frames):
        frames[i] = emphasized[i * FRAME_SHIFT : i * FRAME_SHIFT + FRAME_LENGTH]

    # Hamming window + FFT.
    window = np.hamming(FRAME_LENGTH).astype(np.float32)
    windowed = frames * window
    spectrum = np.fft.rfft(windowed, n=N_FFT)
    power = np.abs(spectrum) ** 2

    # Mel filterbank.
    fbank = _mel_fbank()
    mel = power @ fbank

    # Log.
    mel = np.where(mel == 0, np.finfo(np.float32).eps, mel)
    log_mel = np.log(mel)
    return mx.array(log_mel)


@lru_cache(maxsize=1)
def _load_weights():
    # Lazy import: huggingface_hub only needed when weights are first loaded.
    from huggingface_hub import hf_hub_download  # noqa: PLC0415

    path = hf_hub_download("aufklarer/WeSpeaker-ResNet34-LM-MLX", "model.safetensors")
    weights = {}
    with safe_open(path, framework="np") as f:
        for k in f.keys():  # noqa: SIM118
            weights[k] = mx.array(f.get_tensor(k))
    return weights


def _conv2d(x, weight, bias, stride=1, padding=0):
    """Conv2d with channels-last input.

    x: (B, H, W, C). weight: (O, KH, KW, I). Returns (B, H_out, W_out, O).
    MLX Conv2d expects NHWC input and weight (O, KH, KW, I).
    """
    out_ch, kh, kw, in_ch = weight.shape
    conv = nn.Conv2d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=(kh, kw),
        stride=(stride, stride),
        padding=(padding, padding),
        bias=True,
    )
    conv.weight = weight
    conv.bias = bias
    return conv(x)


def _basic_block(x, prefix, weights, stride=1, has_shortcut=False):
    """ResNet BasicBlock: conv1->relu->conv2 + shortcut -> relu.

    BatchNorm was fused into conv biases at conversion time.
    Shortcut is a 1x1 conv when channels/stride change, else identity.
    """
    identity = x
    out = _conv2d(
        x,
        weights[f"{prefix}.conv1.weight"],
        weights[f"{prefix}.conv1.bias"],
        stride=stride,
        padding=1,
    )
    out = nn.relu(out)
    out = _conv2d(
        out, weights[f"{prefix}.conv2.weight"], weights[f"{prefix}.conv2.bias"], stride=1, padding=1
    )
    if has_shortcut:
        identity = _conv2d(
            x,
            weights[f"{prefix}.shortcut.weight"],
            weights[f"{prefix}.shortcut.bias"],
            stride=stride,
            padding=0,
        )
    out = out + identity
    return nn.relu(out)


def _layer(x, layer_idx, weights, num_blocks, stride):
    """Full ResNet layer. First block has shortcut if stride>1 or channels change."""
    for b in range(num_blocks):
        prefix = f"layer{layer_idx}.{b}"
        is_first = b == 0
        block_stride = stride if is_first else 1
        has_shortcut = is_first and f"{prefix}.shortcut.weight" in weights
        x = _basic_block(x, prefix, weights, stride=block_stride, has_shortcut=has_shortcut)
    return x


def _wespeaker_forward(log_mel, weights):
    """Forward pass. log_mel: (B, T, 80, 1) or (T, 80, 1).

    Returns (B, 256) L2-normalized embeddings.
    """
    if log_mel.ndim == 3:
        log_mel = log_mel[None, :]  # (T, 80, 1) -> (1, T, 80, 1)
    elif log_mel.ndim == 2:
        log_mel = log_mel[None, :, :, None]  # (T, 80) -> (1, T, 80, 1)

    x = _conv2d(log_mel, weights["conv1.weight"], weights["conv1.bias"], stride=1, padding=1)
    x = nn.relu(x)

    x = _layer(x, 1, weights, num_blocks=3, stride=1)
    x = _layer(x, 2, weights, num_blocks=4, stride=2)
    x = _layer(x, 3, weights, num_blocks=6, stride=2)
    x = _layer(x, 4, weights, num_blocks=3, stride=2)

    # Statistics pooling: mean + std across time dimension only.
    # x: (B, T, F, C). Pool over T -> (B, F, C), then flatten + concat.
    mean = x.mean(axis=1)  # (B, F, C)
    std = mx.sqrt(x.var(axis=1) + 1e-5)
    pooled = mx.concatenate([mean, std], axis=-1)  # (B, F, 2C)
    # Batch dim; tensor-shape convention.
    B = pooled.shape[0]  # noqa: N806
    pooled = pooled.reshape(B, -1)  # (B, F * 2C) = (B, 5120)

    # Embedding layer.
    emb = pooled @ weights["embedding.weight"].T + weights["embedding.bias"]

    # L2 normalize.
    norm = mx.sqrt(mx.sum(emb * emb, axis=-1, keepdims=True) + 1e-12)
    return emb / norm


def embed(audio, weights=None):
    """Compute speaker embedding for a raw audio segment.

    audio: mx.array or numpy (samples,) at 16kHz.
    Returns: mx.array (256,) L2-normalized embedding.
    """
    if weights is None:
        weights = _load_weights()
    log_mel = compute_log_mel(audio)
    emb = _wespeaker_forward(log_mel, weights)
    return emb[0]
