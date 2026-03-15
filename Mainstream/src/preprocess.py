"""
preprocess.py
-------------
Xử lý tín hiệu EEG thô từ DEAP dataset:
  - Bandpass filter (theta, alpha, beta, gamma)
  - Baseline removal (trừ trung bình 3 giây đầu)
  - Tính Power Spectral Density (PSD) bằng Welch method
  - Tính Differential Entropy (DE)
"""

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
