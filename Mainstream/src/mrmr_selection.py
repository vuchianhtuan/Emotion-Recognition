"""
mrmr_selection.py
-----------------
MRMR (Minimum Redundancy Maximum Relevance) channel selection for EEG signals.

Upgraded from DEAP-Emotion-Recognition: Use sklearn for MRMR implementation
to ensure compatibility and avoid dependency issues.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import mrmr
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import normalize, StandardScaler

from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import mrmr
from tqdm import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import normalize, StandardScaler

from .preprocess import PreprocessConfig, preprocess_subject_for_mrmr, bin_power_fft


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


# ─────────────────────────── Paths ────────────────────────────────────────── #

RAW_DATA_PATH = Path("data") / "raw"
SAVE_MRMR_CHANNELS_PATH = Path("data") / "saved_trained_mrmr_channels"
FINAL_DATASET_PATH_MRMR = Path("data") / "final_deap_dataset_mrmr"


# ─────────────────────────── Data Loading ─────────────────────────────────── #

def load_subject(participant_id: int) -> dict:
    """Load DEAP subject data from .dat file."""
    filename = f"s{participant_id:02d}.dat"
    filepath = RAW_DATA_PATH / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    with open(filepath, 'rb') as f:
        subject = pickle.load(f, encoding="latin1")
    return subject


def preprocess_subject_global(subject: dict, classify_type: str = "arousal") -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess subject for global MRMR: return (features, labels) for all windows."""
    # Sử dụng preprocess_subject_for_mrmr hoặc implement đơn giản
    cfg = PreprocessConfig()
    # Giả sử preprocess_subject_for_mrmr trả về array (n_windows, 2) với [features, labels]
    # Nhưng cần sửa để trả về (features, labels)

    # Implement đơn giản như trong test_mrmr.py
    meta = []
    for i in range(40):  # 40 trials
        data = subject["data"][i]
        labels = subject["labels"][i][:2]
        start = 0
        while start + WINDOW_SIZE < data.shape[1]:
            meta_array = []
            meta_data = []
            for j in range(N_CHANNELS):
                x = data[j][start: start + WINDOW_SIZE]
                y = bin_power_fft(x, band=BANDS, fs=SAMPLING_RATE)
                meta_data.append(np.array(y[0]))  # list of 5 floats
            meta_array.append(np.stack(meta_data, axis=0))  # (32, 5)
            label_bin = np.array(labels >= LABEL_THRESHOLD).astype(int)
            meta_array.append(label_bin)
            meta.append(np.array(meta_array, dtype=object))
            start += STEP_SIZE

    # Extract features and labels
    n_windows = len(meta)
    features = np.array([meta[i][0] for i in range(n_windows)])  # (n_windows, 32, 5)
    labels = np.array([meta[i][1] for i in range(n_windows)])    # (n_windows, 2)

    # Reshape features to (n_windows, 32*5) or keep as is
    features = features.reshape(n_windows, -1)  # (n_windows, 160)

    # Labels: valence or arousal
    label_idx = 0 if classify_type == "valence" else 1
    labels = labels[:, label_idx]

    return features, labels


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
    """Greedy MRMR channel selection using optimized mrmr library.

    Uses the mrmr library which supports multi-core processing.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        y: 1-D integer label array of shape ``(n_samples,)``.
        K: Number of features to select.

    Returns:
        List of ``K`` selected feature (column) indices.
    """
    # Convert to pandas
    X_df = pd.DataFrame(X)
    y_series = pd.Series(y)
    
    # Use optimized mrmr library with multi-core and progress bar
    selected_features = mrmr.mrmr_classif(
        X_df, y_series, K=K, 
        n_jobs=-1,  # Use all CPU cores
        show_progress=True
    )
    
    # Convert feature names back to indices
    selected_indices = [int(feat) for feat in selected_features]
    return selected_indices


# ─────────────────────────── Global MRMR ──────────────────────────────────── #

def use_mrmr_global(participant_list=range(1, 33), components=20, classify_type: str = "Arousal"):
    """Global MRMR: Select channels once on all participants combined."""
    print(f"Run Global MRMR channel selection with {classify_type} to select {components} channels")

    # Paths
    save_path_data_training = FINAL_DATASET_PATH_MRMR / "data_training.npy"
    save_path_label_training = FINAL_DATASET_PATH_MRMR / "label_training.npy"
    save_path_data_testing = FINAL_DATASET_PATH_MRMR / "data_testing.npy"
    save_path_label_testing = FINAL_DATASET_PATH_MRMR / "label_testing.npy"
    channels_file = SAVE_MRMR_CHANNELS_PATH / f"mrmr_global_channels_{classify_type}.csv"

    FINAL_DATASET_PATH_MRMR.mkdir(exist_ok=True, parents=True)
    SAVE_MRMR_CHANNELS_PATH.mkdir(exist_ok=True, parents=True)

    # 1. Gộp tất cả data từ tất cả participants
    print("Step 1: Loading and combining data from all participants...")
    all_features = []
    all_labels = []

    for participant in tqdm(participant_list, desc="Loading participants"):
        subject = load_subject(participant)
        features, labels = preprocess_subject_global(subject, classify_type.lower())
        all_features.append(features)
        all_labels.append(labels)

    # Gộp
    all_features = np.vstack(all_features)  # (total_windows, 160)
    all_labels = np.concatenate(all_labels)  # (total_windows,)

    print(f"Combined data shape: {all_features.shape}, labels shape: {all_labels.shape}")

    # 2. Chạy MRMR trên toàn bộ data
    print("Step 2: Running MRMR on combined data...")
    selected_indices = _mrmr_classif(all_features, all_labels, K=components)

    # 3. Lưu selected channels
    print(f"Step 3: Saving selected channels to {channels_file}")
    pd.DataFrame({"channels": selected_indices}).to_csv(channels_file, index=False)

    # 4. Áp dụng selected channels cho từng participant
    print("Step 4: Applying selected channels to each participant...")
    x_train = []
    y_train = []
    x_test = []
    y_test = []

    for participant in tqdm(participant_list, desc="Applying channels"):
        subject = load_subject(participant)
        features, labels = preprocess_subject_global(subject, classify_type.lower())

        # Filter channels: giả sử features (n_windows, 160), reshape về (n_windows, 32, 5), filter, flatten lại
        features_reshaped = features.reshape(-1, 32, N_FREQUENCIES)  # (n_windows, 32, 5)
        features_filtered = features_reshaped[:, selected_indices, :]  # (n_windows, components, 5)
        features_filtered = features_filtered.reshape(-1, components * N_FREQUENCIES)  # (n_windows, components*5)

        # Chia train/test
        for i in range(len(features_filtered)):
            if i % TEST_SPLIT_MODULO == 0:
                x_test.append(features_filtered[i])
                y_test.append(labels[i])
            else:
                x_train.append(features_filtered[i])
                y_train.append(labels[i])

    # 5. Lưu dataset
    print("Step 5: Saving filtered dataset...")
    np.save(save_path_data_training, np.array(x_train), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_training, np.array(y_train), allow_pickle=True, fix_imports=True)
    np.save(save_path_data_testing, np.array(x_test), allow_pickle=True, fix_imports=True)
    np.save(save_path_label_testing, np.array(y_test), allow_pickle=True, fix_imports=True)

    print("Global MRMR completed. Dataset saved with selected channels.")


# ─────────────────────────── Legacy per-subject MRMR ─────────────────────── #

def use_mrmr(participant_list=range(1, 33), components=20, classify_type: str = "Arousal"):
    """Legacy: Per-subject MRMR (giữ để tương thích). Now calls global MRMR."""
    use_mrmr_global(participant_list, components, classify_type)


def load_selected_channels(classify_type: str = "arousal") -> List[int]:
    """Load selected channels from global MRMR file."""
    channels_file = SAVE_MRMR_CHANNELS_PATH / f"mrmr_global_channels_{classify_type}.csv"
    if not channels_file.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_file}")
    df = pd.read_csv(channels_file)
    return df["channels"].tolist()


def filter_channels(features: np.ndarray, selected_channels: List[int], n_frequencies: int = N_FREQUENCIES) -> np.ndarray:
    """Filter features to keep only selected channels."""
    # features: (n_windows, 32 * n_frequencies)
    features_reshaped = features.reshape(-1, 32, n_frequencies)  # (n_windows, 32, n_frequencies)
    features_filtered = features_reshaped[:, selected_channels, :]  # (n_windows, len(selected_channels), n_frequencies)
    return features_filtered.reshape(-1, len(selected_channels) * n_frequencies)  # (n_windows, len(selected_channels) * n_frequencies)


def preprocess_and_filter_new_data(subject: dict, selected_channels: List[int], classify_type: str = "arousal") -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess new subject data and filter to selected channels for prediction."""
    features, labels = preprocess_subject_global(subject, classify_type)
    features_filtered = filter_channels(features, selected_channels)
    return features_filtered, labels

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
    cfg = PreprocessConfig(
        fs=fs,
        window_size=window_size,
        overlap=1.0 - (step_size / float(window_size)),
        n_eeg_channels=N_CHANNELS,
        label_threshold=LABEL_THRESHOLD,
        bands=tuple(BANDS if band is None else band),
    )
    return preprocess_subject_for_mrmr(subject, cfg)


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


def run_mrmr_global_selection(
    all_subjects_preprocessed: List[np.ndarray],
    classify_type: str = "arousal",
    K: int = MRMR_COMPONENTS,
) -> List[int]:
    """Run global MRMR channel selection on all preprocessed subjects combined.

    Args:
        all_subjects_preprocessed: List of preprocessed data arrays from preprocess_subject_fft.
        classify_type: "arousal" or "valence".
        K: Number of channels to select.

    Returns:
        selected_channel_indices: List of K channel indices (same for all subjects).
    """
    print(f"Running global MRMR on {len(all_subjects_preprocessed)} subjects, selecting {K} channels for {classify_type}")

    # Combine all data
    all_data = []
    all_labels = []

    for preprocessed in all_subjects_preprocessed:
        n_windows = preprocessed.shape[0]
        data_list = [preprocessed[i][0] for i in range(n_windows)]   # (N_CH, N_FREQ)
        label_list = [preprocessed[i][1] for i in range(n_windows)]  # [valence_bin, arousal_bin]

        data = np.array(data_list)    # (n_windows, N_CH, N_FREQ)
        labels = np.array(label_list)  # (n_windows, 2)

        # Reshape to (n_windows * N_FREQ, N_CH)
        n_ch = data.shape[1]
        x = data.transpose((1, 0, 2)).reshape(n_ch, -1).transpose((1, 0))

        # Labels: repeat for each frequency
        if classify_type.lower() == "arousal":
            y = np.repeat(labels[:, 0], N_FREQUENCIES)
        else:
            y = np.repeat(labels[:, 1], N_FREQUENCIES)

        all_data.append(x)
        all_labels.append(y)

    # Stack all
    all_data = np.vstack(all_data)  # (total_windows, N_CH)
    all_labels = np.concatenate(all_labels)  # (total_windows,)

    print(f"Combined data shape: {all_data.shape}, labels shape: {all_labels.shape}")

    # Run MRMR once on combined data
    selected = _mrmr_classif(all_data, all_labels.astype(int), K=K)
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