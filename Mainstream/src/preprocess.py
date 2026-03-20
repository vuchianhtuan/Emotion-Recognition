"""
preprocess.py
-------------
Xử lý tín hiệu EEG thô từ DEAP dataset:
  - Band-pass + notch filter (MNE-Python)
  - Baseline correction
  - Artifact handling cơ bản theo z-score
  - Tính Power Spectral Density (PSD) bằng Welch method
  - Tính Differential Entropy (DE)
  - Tính FFT band-power 5 dải (Theta/Alpha/LowerBeta/UpperBeta/Gamma)
    dùng cho pipeline MRMR – thay thế pyeeg.bin_power bằng numpy FFT
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from scipy.signal import welch, butter, sosfilt


# ----------- Constants -----------
SAMPLING_RATE = 128  # Hz – DEAP dataset
BANDS = {
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}
BASELINE_SECONDS = 3  # Số giây đầu dùng làm baseline

# 5-band edges dùng cho pipeline MRMR (khớp với DEAP-Emotion-Recognition gốc)
MRMR_BANDS = [4, 8, 12, 16, 25, 45]   # Theta / Alpha / LowerBeta / UpperBeta / Gamma
MRMR_BAND_NAMES = ["Theta", "Alpha", "LowerBeta", "UpperBeta", "Gamma"]

PIPELINE_VERSION = "v2_mne"


@dataclass(frozen=True)
class PreprocessConfig:
    """Cấu hình preprocess chuẩn cho DEAP."""

    fs: int = SAMPLING_RATE
    l_freq: float = 4.0
    h_freq: float = 45.0
    notch_freq: float = 50.0
    baseline_seconds: float = BASELINE_SECONDS
    artifact_zscore: float = 5.0
    window_size: int = 256
    overlap: float = 0.9375  # 240/256 -> step=16 như pipeline gốc
    n_eeg_channels: int = 32
    label_threshold: float = 5.0
    bands: tuple[int, ...] = tuple(MRMR_BANDS)
    pipeline_version: str = PIPELINE_VERSION

    @property
    def step_size(self) -> int:
        step = int(round(self.window_size * (1.0 - self.overlap)))
        return max(step, 1)


def _eeg_channel_names(n_channels: int) -> list[str]:
    return [f"EEG{idx + 1:02d}" for idx in range(n_channels)]


def clean_eeg_trial_mne(
    trial: np.ndarray,
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Làm sạch trial EEG bằng MNE: band-pass + notch + baseline + artifact clamp."""
    cfg = config or PreprocessConfig()
    trial = np.asarray(trial, dtype=np.float64)

    info = mne.create_info(
        ch_names=_eeg_channel_names(trial.shape[0]),
        sfreq=cfg.fs,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(trial, info, verbose="ERROR")
    raw.filter(l_freq=cfg.l_freq, h_freq=cfg.h_freq, verbose="ERROR")

    if cfg.notch_freq > 0 and cfg.notch_freq < (cfg.fs / 2.0):
        raw.notch_filter(freqs=[cfg.notch_freq], verbose="ERROR")

    data = raw.get_data()

    baseline_samples = max(1, int(round(cfg.baseline_seconds * cfg.fs)))
    baseline_samples = min(baseline_samples, data.shape[1])
    baseline = data[:, :baseline_samples].mean(axis=1, keepdims=True)
    data = data - baseline

    # Artifact handling cơ bản: clamp theo z-score mỗi kênh.
    ch_mean = data.mean(axis=1, keepdims=True)
    ch_std = data.std(axis=1, keepdims=True) + 1e-10
    z = (data - ch_mean) / ch_std
    med = np.median(data, axis=1, keepdims=True)
    data = np.where(np.abs(z) > cfg.artifact_zscore, med, data)

    return data.astype(np.float32)


# ----------- Filtering -----------
def bandpass_filter(signal: np.ndarray, low: float, high: float, fs: int = SAMPLING_RATE) -> np.ndarray:
    """Áp dụng Butterworth bandpass filter cho tín hiệu 1-D hoặc (channels, samples)."""
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    if signal.ndim == 1:
        return sosfilt(sos, signal)
    return np.array([sosfilt(sos, ch) for ch in signal])


# ----------- Baseline Removal -----------
def remove_baseline(trial: np.ndarray, fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Trừ baseline (trung bình của `BASELINE_SECONDS` giây đầu) khỏi toàn bộ trial.
    trial shape: (channels, samples)
    """
    baseline_samples = BASELINE_SECONDS * fs
    baseline = trial[:, :baseline_samples].mean(axis=1, keepdims=True)
    return trial - baseline


# ----------- PSD (Welch) -----------
def compute_psd(signal: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """
    Tính PSD trung bình trong mỗi dải tần số cho 1 kênh.
    signal: 1-D array (samples,)
    Trả về dict {band_name: psd_mean}
    """
    freqs, psd = welch(signal, fs=fs, nperseg=fs * 2)
    result = {}
    for band, (low, high) in BANDS.items():
        idx = np.logical_and(freqs >= low, freqs <= high)
        result[band] = psd[idx].mean()
    return result


# ----------- Differential Entropy -----------
def compute_de(signal: np.ndarray, fs: int = SAMPLING_RATE) -> dict:
    """
    Tính Differential Entropy: DE = 0.5 * log(2πe * σ²)
    Tính trên từng dải tần.
    signal: 1-D array (samples,)
    """
    result = {}
    for band, (low, high) in BANDS.items():
        filtered = bandpass_filter(signal, low, high, fs)
        variance = np.var(filtered)
        de = 0.5 * np.log(2 * np.pi * np.e * (variance + 1e-10))
        result[band] = de
    return result


# ----------- Full Pipeline -----------
def extract_features(trial: np.ndarray, mode: str = "psd", fs: int = SAMPLING_RATE) -> np.ndarray:
    """
    Pipeline đầy đủ: baseline removal → tính đặc trưng trên từng channel.

    Args:
        trial: (channels, samples)
        mode : "psd" hoặc "de"

    Returns:
        features: (channels, n_bands)
    """
    trial = remove_baseline(trial, fs)
    compute_fn = compute_psd if mode == "psd" else compute_de

    features = []
    for ch in trial:
        band_feats = compute_fn(ch, fs)
        features.append(list(band_feats.values()))
    return np.array(features, dtype=np.float32)


# ----------- FFT 5-band power (MRMR pipeline) -----------

def bin_power_fft(
    x: np.ndarray,
    band: list = None,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """Compute mean FFT power per frequency band for a 1-D signal.

    Replaces ``pyeeg.bin_power`` from the original DEAP-Emotion-Recognition code.

    Args:
        x   : 1-D EEG signal.
        band: Band-edge list, e.g. ``[4, 8, 12, 16, 25, 45]``.
              Defaults to ``MRMR_BANDS``.
        fs  : Sampling rate in Hz.

    Returns:
        powers: Array of shape ``(len(band) - 1,)`` — mean FFT power per band.
    """
    if band is None:
        band = MRMR_BANDS

    n = len(x)
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    powers = []
    for i in range(len(band) - 1):
        low, high = band[i], band[i + 1]
        idx = np.where((freqs >= low) & (freqs < high))[0]
        powers.append(float(np.mean(fft_vals[idx])) if len(idx) > 0 else 0.0)

    return np.array(powers, dtype=np.float32)


def extract_fft_features(
    trial: np.ndarray,
    band: list = None,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """Compute FFT band-power for all EEG channels of one trial window.

    Used by the MRMR pipeline (5-band version matching the original DEAP code).

    Args:
        trial: 2-D array of shape ``(n_channels, n_samples)``.
        band : Band-edge list. Defaults to ``MRMR_BANDS``.
        fs   : Sampling rate in Hz.

    Returns:
        features: Array of shape ``(n_channels, n_bands)`` — float32.
    """
    if band is None:
        band = MRMR_BANDS

    features = []
    for ch in trial:
        features.append(bin_power_fft(ch, band, fs))
    return np.array(features, dtype=np.float32)


def preprocess_subject_for_mrmr(
    subject: dict[str, Any],
    config: PreprocessConfig | None = None,
) -> np.ndarray:
    """Preprocess một subject DEAP thành meta array tương thích MRMR hiện tại."""
    cfg = config or PreprocessConfig()
    n_trials = int(subject["data"].shape[0])
    meta = []

    for trial_idx in range(n_trials):
        trial = np.asarray(subject["data"][trial_idx], dtype=np.float32)[: cfg.n_eeg_channels]
        labels = np.asarray(subject["labels"][trial_idx][:2], dtype=np.float32)
        label_bin = (labels >= cfg.label_threshold).astype(np.int32)

        cleaned = clean_eeg_trial_mne(trial, cfg)
        start = 0
        while start + cfg.window_size < cleaned.shape[1]:
            window = cleaned[:, start: start + cfg.window_size]
            feats = extract_fft_features(window, band=list(cfg.bands), fs=cfg.fs)
            meta.append(np.array([feats, label_bin], dtype=object))
            start += cfg.step_size

    return np.array(meta, dtype=object)


def save_processed_subject(
    meta: np.ndarray,
    output_dir: str | Path,
    subject_id: str,
    config: PreprocessConfig | None = None,
) -> Path:
    """Lưu output preprocess đã version hóa vào data/processed."""
    cfg = config or PreprocessConfig()
    output_path = Path(output_dir) / cfg.pipeline_version
    output_path.mkdir(parents=True, exist_ok=True)
    target = output_path / f"{subject_id}.npz"

    features = np.stack([meta[i][0] for i in range(meta.shape[0])]).astype(np.float32)
    labels = np.stack([meta[i][1] for i in range(meta.shape[0])]).astype(np.int32)

    np.savez_compressed(
        target,
        features=features,
        labels=labels,
        config_json=json.dumps(asdict(cfg)),
    )
    return target


def run_preprocess_cli(args: argparse.Namespace) -> None:
    cfg = PreprocessConfig(
        fs=args.fs,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        notch_freq=args.notch_freq,
        baseline_seconds=args.baseline_seconds,
        artifact_zscore=args.artifact_zscore,
        window_size=args.window_size,
        overlap=args.overlap,
        n_eeg_channels=args.n_channels,
        pipeline_version=args.version,
    )
    dat_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {args.data_dir}")

    for fname in dat_files:
        path = Path(args.data_dir) / fname
        with open(path, "rb") as f:
            subject = pickle.load(f, encoding="latin1")
        meta = preprocess_subject_for_mrmr(subject, cfg)
        saved = save_processed_subject(meta, args.output_dir, subject_id=Path(fname).stem, config=cfg)
        print(f"[OK] {fname}: {meta.shape[0]} windows -> {saved}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DEAP preprocessing pipeline (MNE + FFT bands)")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--version", default=PIPELINE_VERSION)
    parser.add_argument("--fs", type=int, default=SAMPLING_RATE)
    parser.add_argument("--l-freq", type=float, default=4.0)
    parser.add_argument("--h-freq", type=float, default=45.0)
    parser.add_argument("--notch-freq", type=float, default=50.0)
    parser.add_argument("--baseline-seconds", type=float, default=float(BASELINE_SECONDS))
    parser.add_argument("--artifact-zscore", type=float, default=5.0)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--overlap", type=float, default=0.9375)
    parser.add_argument("--n-channels", type=int, default=32)
    return parser


if __name__ == "__main__":
    run_preprocess_cli(build_argparser().parse_args())
