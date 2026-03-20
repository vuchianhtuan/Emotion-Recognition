"""
mrmr_selection.py
-----------------
MRMR (Minimum Redundancy Maximum Relevance) channel selection for EEG signals.

This module is a modern Python conversion of:
  DEAP-Emotion-Recognition/FeatureExtraction/MRMR.py

Key changes from original:
  - pyeeg.bin_power replaced by numpy FFT (bin_power_fft)
  - MRMR implemented using scikit-learn mutual_info_classif
    (replaces the external mrmr-py dependency for wider compatibility)
  - Compatible with PyTorch-based training pipeline in Mainstream

Pipeline:
  1. Sliding-window FFT on each DEAP .dat subject
  2. MRMR channel selection per subject
  3. Build flattened train / test arrays
  4. Normalize for LSTM input
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler, normalize


# ─────────────────────────── Constants ────────────────────────────────────── #

BANDS: List[int] = [4, 8, 12, 16, 25, 45]   # Band edge frequencies in Hz
SAMPLING_RATE: int = 128                      # Hz (DEAP dataset)
N_CHANNELS: int = 32                          # EEG channels
N_FREQUENCIES: int = len(BANDS) - 1          # 5 frequency bands
WINDOW_SIZE: int = 256                        # Samples per window (2 s × 128 Hz)
STEP_SIZE: int = 16                           # Sliding step
MRMR_COMPONENTS: int = 20                    # Default top-K channels
LABEL_THRESHOLD: int = 5                     # Valence/Arousal score ≥ threshold → High
TEST_SPLIT_MODULO: int = 4                   # Every N-th window goes to test set (25%)


# ───────────────────── FFT band-power (pyeeg replacement) ─────────────────── #

def bin_power_fft(
    x: np.ndarray,
    band: List[int] = BANDS,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """Compute mean FFT power in each frequency band for a 1-D signal.

    Replaces ``pyeeg.bin_power`` with numpy FFT.

    Args:
        x   : 1-D EEG signal.
        band: List of band-edge frequencies, e.g. [4, 8, 12, 16, 25, 45].
        fs  : Sampling rate in Hz.

    Returns:
        powers: Array of shape ``(len(band) - 1,)`` with mean power per band.
    """
    n = len(x)
    fft_vals = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    powers = []
    for i in range(len(band) - 1):
        low, high = band[i], band[i + 1]
        idx = np.where((freqs >= low) & (freqs < high))[0]
        powers.append(float(np.mean(fft_vals[idx])) if len(idx) > 0 else 0.0)

    return np.array(powers, dtype=np.float32)


# ───────────────────── MRMR implementation ────────────────────────────────── #

def _mrmr_classif(X: np.ndarray, y: np.ndarray, K: int) -> List[int]:
    """Greedy MRMR channel selection using scikit-learn mutual information.

    At each step selects the feature ``f`` that maximises::

        score(f) = MI(f, y) - (1 / |S|) * sum(MI(f, s) for s in S)

    where ``S`` is the already-selected feature set.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: 1-D integer label array of shape ``(n_samples,)``.
        K: Number of features to select.

    Returns:
        List of ``K`` selected feature (column) indices.
    """
    from sklearn.feature_selection import mutual_info_regression

    n_features = X.shape[1]
    K = min(K, n_features)

    # Relevance: MI(feature_i, target) — target is discrete class label
    relevance = mutual_info_classif(X, y, discrete_features=False, random_state=42)

    selected: List[int] = []
    remaining = list(range(n_features))

    for _ in range(K):
        if not selected:
            # First feature: pick highest relevance
            best = int(np.argmax(relevance))
            selected.append(best)
            remaining.remove(best)
            continue

        # Redundancy: average MI between candidate and already-selected features
        # Feature-feature MI uses mutual_info_regression (both continuous)
        best_score = -np.inf
        best_feat = remaining[0]

        for feat in remaining:
            red = np.mean([
                mutual_info_regression(
                    X[:, [feat]], X[:, s],
                    discrete_features=False, random_state=42,
                ).item()
                for s in selected
            ])
            score = relevance[feat] - red
            if score > best_score:
                best_score = score
                best_feat = feat

        selected.append(best_feat)
        remaining.remove(best_feat)

    return selected


# ───────────────────── Per-subject FFT preprocessing ─────────────────────── #

def preprocess_subject_fft(
    subject: dict,
    band: Optional[List[int]] = None,
    window_size: int = WINDOW_SIZE,
    step_size: int = STEP_SIZE,
    fs: int = SAMPLING_RATE,
) -> np.ndarray:
    """Apply sliding-window FFT on a DEAP subject dictionary.

    Args:
        subject    : Dictionary loaded from a DEAP ``.dat`` file with keys
                     ``"data"`` (40, 40, 8064) and ``"labels"`` (40, 4).
        band       : Band-edge list. Defaults to ``BANDS``.
        window_size: Samples per window.
        step_size  : Sliding step in samples.
        fs         : Sampling rate in Hz.

    Returns:
        meta: Array of shape ``(n_windows,)`` with dtype ``object``.
              Each element is ``[features, label_bin]`` where
              ``features`` has shape ``(N_CHANNELS, N_FREQUENCIES)`` and
              ``label_bin`` is ``[valence_bin, arousal_bin]`` — matching the
              DEAP label order used by ``PreProcessing/FFT.py`` (col 0 = valence,
              col 1 = arousal).
    """
    if band is None:
        band = BANDS

    meta = []
    for trial_idx in range(40):
        data = subject["data"][trial_idx]       # (40_ch, 8064_samples)
        # DEAP labels order: [valence, arousal, dominance, liking]
        # Take first two → [valence, arousal]; matches PreProcessing/FFT.py
        labels = subject["labels"][trial_idx][:2]  # [valence, arousal]

        start = 0
        while start + window_size < data.shape[1]:
            meta_data = []
            for ch in range(N_CHANNELS):
                x = data[ch][start: start + window_size]
                meta_data.append(bin_power_fft(x, band, fs))  # (N_FREQ,)

            meta_array = np.array(meta_data)          # (N_CHANNELS, N_FREQ)
            label_bin = (labels >= LABEL_THRESHOLD).astype(int)
            meta.append(np.array([meta_array, label_bin], dtype=object))
            start += step_size

    return np.array(meta, dtype=object)


# ─────────────────────────── MRMR selection ───────────────────────────────── #

def run_mrmr_selection(
    preprocessed_data: np.ndarray,
    classify_type: str = "arousal",
    K: int = MRMR_COMPONENTS,
) -> List[int]:
    """Run MRMR channel selection on preprocessed FFT data for one subject.

    Mirrors the per-subject MRMR step in the original DEAP code.

    Args:
        preprocessed_data: Array returned by :func:`preprocess_subject_fft`.
        classify_type    : ``"arousal"`` or ``"valence"``.
        K                : Number of channels to select.

    Returns:
        selected_channel_indices: List of ``K`` integer channel indices.
    """
    n_windows = preprocessed_data.shape[0]

    data_list = [preprocessed_data[i][0] for i in range(n_windows)]   # (N_CH, N_FREQ)
    label_list = [preprocessed_data[i][1] for i in range(n_windows)]  # [valence_bin, arousal_bin]

    data = np.array(data_list)    # (n_windows, N_CH, N_FREQ)
    labels = np.array(label_list)  # (n_windows, 2)  col 0=valence, col 1=arousal

    # Reshape to (n_windows * N_FREQ, N_CH) — each row is one time-frequency frame.
    # Use data.shape[1] (actual channel count) instead of the N_CHANNELS constant
    # to avoid incorrect reshape if the data has a non-default number of channels.
    n_ch = data.shape[1]
    x = data.transpose((1, 0, 2)).reshape(n_ch, -1).transpose((1, 0))

    # Mirrors FeatureExtraction/MRMR.py: col 0 for "arousal", col 1 for "valence".
    # Note: DEAP labels are stored as [valence, arousal], so col 0 is valence and
    # col 1 is arousal — this intentionally reproduces the original DEAP column
    # selection to keep selection behavior aligned with DEAP-Emotion-Recognition.
    if classify_type.lower() == "arousal":
        y = np.repeat(labels[:, 0], N_FREQUENCIES)
    else:
        y = np.repeat(labels[:, 1], N_FREQUENCIES)

    selected = _mrmr_classif(x, y.astype(int), K=K)
    return selected


# ─────────────────── Dataset builder from MRMR-selected channels ──────────── #

def build_mrmr_dataset(
    all_subjects_data: List[np.ndarray],
    selected_channels_per_subject: List[List[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build flattened train / test arrays using MRMR-selected channels.

    Args:
        all_subjects_data            : List of arrays from
                                       :func:`preprocess_subject_fft`.
        selected_channels_per_subject: List of channel index lists (one per subject).

    Returns:
        x_train, y_train, x_test, y_test — raw (unscaled) numpy arrays.
        Each sample in x is shape ``(K * N_FREQ,)``.
        Each label in y is shape ``(2,)`` — ``[valence_bin, arousal_bin]``
        (col 0 = valence, col 1 = arousal; matches DEAP label storage order).
    """
    x_train, y_train = [], []
    x_test, y_test = [], []

    for sub_data, selected_channels in zip(all_subjects_data, selected_channels_per_subject):
        n_windows = sub_data.shape[0]
        data_list = [sub_data[i][0] for i in range(n_windows)]
        label_list = [sub_data[i][1] for i in range(n_windows)]

        data = np.array(data_list)    # (n_windows, N_CH, N_FREQ)
        labels = np.array(label_list)  # (n_windows, 2)  col 0=valence, col 1=arousal

        # Flatten to (n_windows * N_FREQ, N_CH) — mirrors FeatureExtraction/MRMR.py.
        # Use data.shape[1] (actual channel count) instead of N_CHANNELS constant
        # to guard against mismatched preprocessing configurations.
        n_ch = data.shape[1]
        x_all = data.transpose((1, 0, 2)).reshape(n_ch, -1).transpose((1, 0))
        x_df = pd.DataFrame(x_all)
        data_new = x_df[selected_channels].to_numpy()  # (n_windows*N_FREQ, K)

        # Reshape each selected channel back to (n_windows, N_FREQ)
        z = []
        for ch_data in data_new.T:
            z.append(ch_data.reshape(-1, N_FREQUENCIES))

        zx = np.array(z).transpose((1, 0, 2))  # (n_windows, K, N_FREQ)

        for i in range(zx.shape[0]):
            flat = zx[i].reshape(-1)  # (K * N_FREQ,)
            if i % TEST_SPLIT_MODULO == 0:
                x_test.append(flat)
                y_test.append(labels[i])
            else:
                x_train.append(flat)
                y_train.append(labels[i])

    return (
        np.array(x_train, dtype=np.float32),
        np.array(y_train, dtype=np.int32),
        np.array(x_test, dtype=np.float32),
        np.array(y_test, dtype=np.int32),
    )


# ─────────────────────── Normalization / reshape helper ──────────────────── #

def prepare_for_lstm(
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    classify_type: str = "arousal",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features and select the target label column.

    Mirrors ``LSTMModel/PrepareDataset.py`` from the original repo
    but fixes the data-leakage bug (uses ``fit_transform`` only on train,
    then ``transform`` on test).

    Args:
        x_train, x_test    : Raw feature arrays.
        y_train, y_test    : Label arrays of shape ``(n, 2)``.
        classify_type      : ``"arousal"`` or ``"valence"``.

    Returns:
        x_train, y_train_bin, x_test, y_test_bin reshaped for LSTM:
        x shape: ``(n, features, 1)``; y shape: ``(n,)`` binary int.
    """
    # L2 normalise rows
    x_train = normalize(x_train).astype(np.float32)
    x_test = normalize(x_test).astype(np.float32)

    # StandardScaler (fit on train only)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    # Reshape → (n, seq_len, 1)
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Select label column.
    # DEAP stores labels as [valence, arousal] → col 0 = valence, col 1 = arousal.
    # FeatureExtraction/MRMR.py and PrepareDataset.py both use col 0 for "Arousal"
    # and col 1 for "Valence" — we reproduce that convention here for full alignment.
    col = 0 if classify_type.lower() == "arousal" else 1
    y_train_bin = y_train[:, col]
    y_test_bin = y_test[:, col]

    return x_train, y_train_bin, x_test, y_test_bin
