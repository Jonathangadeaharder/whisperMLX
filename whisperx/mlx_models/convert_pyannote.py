"""Convert the pyannote segmentation checkpoint to MLX-compatible safetensors.

Loads whisperx/assets/pytorch_model.bin (the Pyannote PyanNet checkpoint),
pre-computes the SincNet ParamSincFB bandpass filters, merges LSTM bias terms
(bias_ih + bias_hh), and saves a complete safetensors file that the MLX
forward pass can load directly.

Run once: python -m whisperx.mlx_models.convert_pyannote
Output: whisperx/assets/pyannote_segmentation_mlx.safetensors
"""

import os

import numpy as np
import torch


def to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def compute_sincnet_filters(low_hz_, band_hz_, window_, n_,
                            min_low_hz=50, min_band_hz=50,
                            sample_rate=16000.0, kernel_size=251):
    """Compute 80 bandpass filters from ParamSincFB parameters.

    low_hz_: (40, 1) learned low cutoff frequencies.
    band_hz_: (40, 1) learned bandwidths.
    window_: (125,) half Hamming window.
    n_: (1, 125) half time vector (2*pi*arange(-half, 0) / sample_rate).

    Returns: (80, 1, 251) bandpass filters (40 cos + 40 sin).
    """
    half_kernel = kernel_size // 2  # 125
    low = min_low_hz + np.abs(low_hz_)
    high = np.clip(
        low + min_band_hz + np.abs(band_hz_),
        min_low_hz, sample_rate / 2,
    )

    def make_filters(low, high, filt_type):
        band = (high - low)[:, 0]
        ft_low = low @ n_
        ft_high = high @ n_
        if filt_type == "cos":
            bp_left = ((np.sin(ft_high) - np.sin(ft_low)) / (n_ / 2)) * window_
            bp_center = 2 * band[:, None]
            bp_right = np.flip(bp_left, axis=1)
        else:  # sin
            bp_left = ((np.cos(ft_low) - np.cos(ft_high)) / (n_ / 2)) * window_
            bp_center = np.zeros_like(band[:, None])
            bp_right = -np.flip(bp_left, axis=1)
        band_pass = np.concatenate([bp_left, bp_center, bp_right], axis=1)
        band_pass = band_pass / (2 * band[:, None])
        return band_pass.reshape(40, 1, kernel_size)

    cos_filters = make_filters(low, high, "cos")
    sin_filters = make_filters(low, high, "sin")
    return np.concatenate([cos_filters, sin_filters], axis=0)


def convert():
    ckpt_path = os.path.join(
        os.path.dirname(__file__), "..", "assets", "pytorch_model.bin"
    )
    ckpt_path = os.path.abspath(ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    out = {}

    # SincNet pre-computed filters (80, 1, 251) -> (80, 251, 1) for MLX Conv1d.
    low_hz_ = ckpt["sincnet.conv1d.0.filterbank.low_hz_"].numpy()
    band_hz_ = ckpt["sincnet.conv1d.0.filterbank.band_hz_"].numpy()
    window_ = ckpt["sincnet.conv1d.0.filterbank.window_"].numpy()
    n_ = ckpt["sincnet.conv1d.0.filterbank.n_"].numpy()
    filters = compute_sincnet_filters(low_hz_, band_hz_, window_, n_)
    # MLX Conv1d weight layout: (out, kernel, in). filters is (80, 1, 251).
    out["sincnet.conv.0.weight"] = np.transpose(filters, (0, 2, 1))

    # SincNet conv1d 1 and 2: (out, in, kernel) -> (out, kernel, in) for MLX.
    for i in [1, 2]:
        w = ckpt[f"sincnet.conv1d.{i}.weight"].numpy()
        b = ckpt[f"sincnet.conv1d.{i}.bias"].numpy()
        out[f"sincnet.conv.{i}.weight"] = np.transpose(w, (0, 2, 1))
        out[f"sincnet.conv.{i}.bias"] = b

    # SincNet InstanceNorm weights (affine, no running stats).
    for i in range(3):
        out[f"sincnet.norm.{i}.weight"] = ckpt[f"sincnet.norm1d.{i}.weight"].numpy()
        out[f"sincnet.norm.{i}.bias"] = ckpt[f"sincnet.norm1d.{i}.bias"].numpy()
    out["sincnet.wav_norm.weight"] = ckpt["sincnet.wav_norm1d.weight"].numpy()
    out["sincnet.wav_norm.bias"] = ckpt["sincnet.wav_norm1d.bias"].numpy()

    # BiLSTM: merge bias_ih + bias_hh for each direction and layer.
    for direction in ["", "_reverse"]:
        for layer in range(4):
            suffix = f"{direction}" if direction == "" else "_reverse"
            prefix = f"lstm_{'fwd' if direction == '' else 'bwd'}.layers.{layer}"
            wih = ckpt[f"lstm.weight_ih_l{layer}{suffix}"].numpy()
            whh = ckpt[f"lstm.weight_hh_l{layer}{suffix}"].numpy()
            bih = ckpt[f"lstm.bias_ih_l{layer}{suffix}"].numpy()
            bhh = ckpt[f"lstm.bias_hh_l{layer}{suffix}"].numpy()
            out[f"{prefix}.Wx"] = wih
            out[f"{prefix}.Wh"] = whh
            out[f"{prefix}.bias"] = bih + bhh

    # Linear layers.
    out["linear.0.weight"] = ckpt["linear.0.weight"].numpy()
    out["linear.0.bias"] = ckpt["linear.0.bias"].numpy()
    out["linear.1.weight"] = ckpt["linear.1.weight"].numpy()
    out["linear.1.bias"] = ckpt["linear.1.bias"].numpy()

    # Classifier: (3, 128) -> outputs 3 classes.
    out["classifier.weight"] = ckpt["classifier.weight"].numpy()
    out["classifier.bias"] = ckpt["classifier.bias"].numpy()

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "assets",
        "pyannote_segmentation_mlx.safetensors",
    )
    out_path = os.path.abspath(out_path)

    from safetensors.numpy import save_file
    save_file({k: v.astype(np.float32) for k, v in out.items()}, out_path)
    print(f"Saved {len(out)} tensors to {out_path}")
    for k, v in sorted(out.items()):
        print(f"  {k}: {v.shape}")


if __name__ == "__main__":
    convert()
