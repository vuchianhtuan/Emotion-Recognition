"""
test_mrmr.py
-------------
Test script để chạy MRMR channel selection trên dữ liệu DEAP participant 1.
Hiển thị kênh nào được chọn và kênh nào bị bỏ.
"""

import os
import pickle
import numpy as np
from pathlib import Path
import time
from tqdm import tqdm
import traceback

# Import từ Mainstream
import sys
sys.path.append(str(Path(__file__).parent))

from mrmr_selection import _mrmr_classif
from preprocess import bin_power_fft, PreprocessConfig

# Constants for MRMR
BANDS = [4, 8]  # Only theta band for testing
SAMPLING_RATE = 128
N_CHANNELS = 32
N_FREQUENCIES = len(BANDS) - 1  # 1
WINDOW_SIZE = 256
STEP_SIZE = 16
LABEL_THRESHOLD = 5.0

def preprocess_subject_fft(subject):
    """Preprocess subject data for MRMR - copy from mrmr_selection.py"""
    meta = []
    for i in range(40):  # 40 trials
        data = subject["data"][i]
        labels = subject["labels"][i][:2]  # valence, arousal
        start = 0

        while start + WINDOW_SIZE < data.shape[1]:
            meta_array = []
            meta_data = []
            for j in range(N_CHANNELS):  # 32 channels
                x = data[j][start: start + WINDOW_SIZE]
                y = bin_power_fft(x, band=BANDS, fs=SAMPLING_RATE)
                meta_data.append(np.array(y[0]))  # powers only

            meta_array.append(np.stack(meta_data, axis=0))  # (32, 5)
            label_bin = np.array(labels >= LABEL_THRESHOLD).astype(int)
            meta_array.append(label_bin)

            meta.append(np.array(meta_array, dtype=object))
            start = start + STEP_SIZE

    return np.array(meta)

# Constants
DEAP_ELECTRODES = ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz",
                   "Pz", "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
                   "PO4", "O2"]

# Đường dẫn đến dữ liệu DEAP (giả sử ở thư mục data/raw)
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
if not DATA_DIR.exists():
    # Nếu không có, dùng đường dẫn từ DEAP-Emotion-Recognition
    DATA_DIR = Path(__file__).parent.parent.parent / "DEAP-Emotion-Recognition" / "Data" / "RAW_DEAP_DATASET"

def load_participant_data(participant_id: int):
    """Load dữ liệu của một participant từ file .dat"""
    filename = f"s{participant_id:02d}.dat"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {filepath}")

    with open(filepath, 'rb') as f:
        subject = pickle.load(f, encoding="latin1")

    return subject

def test_mrmr_on_participant(participant_id: int = 1, target: str = "arousal", k: int = 20):
    """Test MRMR trên một participant"""
    print(f"Testing MRMR on participant {participant_id}, target: {target}, selecting {k} channels")

    start_time = time.time()

    # 1. Load dữ liệu thô
    print("Step 1: Loading participant data...")
    load_start = time.time()
    subject = load_participant_data(participant_id)
    load_time = time.time() - load_start
    print(".2f")

    # 2. Preprocess theo MRMR pipeline
    print("Step 2: Preprocessing EEG data (FFT + windowing)...")
    preprocess_start = time.time()
    preprocessed_data = preprocess_subject_fft(subject)
    preprocess_time = time.time() - preprocess_start
    print(".2f")

    print(f"Preprocessed data shape: {preprocessed_data.shape}")

    # 3. Extract features and labels như trong run_mrmr_selection
    print("Step 3: Extracting features and labels...")
    extract_start = time.time()
    n_windows = preprocessed_data.shape[0]

    print(f"Processing {n_windows} windows...")
    data_list = []
    label_list = []
    for i in tqdm(range(n_windows), desc="Extracting features", unit="window"):
        data_list.append(preprocessed_data[i][0])   # (32, 5)
        label_list.append(preprocessed_data[i][1])  # [valence_bin, arousal_bin]

    # Debug: check shapes
    if data_list:
        print(f"Sample data_list[0].shape: {data_list[0].shape}")
        print(f"Sample label_list[0].shape: {label_list[0].shape}")
        unique_data_shapes = set(d.shape for d in data_list[:100])  # check first 100
        unique_label_shapes = set(l.shape for l in label_list[:100])
        print(f"Unique data shapes (first 100): {unique_data_shapes}")
        print(f"Unique label shapes (first 100): {unique_label_shapes}")

    try:
        data = np.stack(data_list, axis=0)    # (n_windows, 32, 5)
        labels = np.stack(label_list, axis=0)  # (n_windows, 2)
    except ValueError as e:
        print(f"Error stacking arrays: {e}")
        # Check all shapes
        data_shapes = [d.shape for d in data_list]
        label_shapes = [l.shape for l in label_list]
        print(f"Data shapes count: {len(set(data_shapes))}")
        print(f"Label shapes count: {len(set(label_shapes))}")
        print(f"First 10 data shapes: {data_shapes[:10]}")
        print(f"First 10 label shapes: {label_shapes[:10]}")
        raise

    print(f"Data shape after stack: {data.shape}")
    print(f"Labels shape after stack: {labels.shape}")

    # For N_FREQUENCIES=1, data.shape = (n_windows, 32), no need to reshape
    x = data

    print(f"X shape: {x.shape}")

    # Handle NaN and Inf values that may be in EEG data
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Labels: col 0 for valence, col 1 for arousal
    y = labels[:, 0 if target == "valence" else 1]  # No repeat since N_FREQUENCIES=1

    extract_time = time.time() - extract_start
    print(".2f")

    print(f"Features shape: {x.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Label distribution: {np.bincount(y)} (0=Low, 1=High)")

    # 4. Chạy MRMR
    print(f"Step 4: Running MRMR channel selection (K={k})...")
    mrmr_start = time.time()
    try:
        selected_indices = _mrmr_classif(x, y, K=k)
    except Exception as e:
        print(f"Error in MRMR: {e}")
        print(f"X shape: {x.shape}, y shape: {y.shape}")
        raise
    mrmr_time = time.time() - mrmr_start
    print(".2f")

    total_time = time.time() - start_time
    print(".2f")

    # 5. Hiển thị kết quả
    print(f"\nSelected {len(selected_indices)} channels (indices): {selected_indices}")

    selected_channels = [DEAP_ELECTRODES[i] for i in selected_indices]
    print(f"Selected channels (names): {selected_channels}")

    # Kênh bị bỏ
    all_indices = set(range(32))
    discarded_indices = sorted(all_indices - set(selected_indices))
    discarded_channels = [DEAP_ELECTRODES[i] for i in discarded_indices]

    print(f"\nDiscarded {len(discarded_indices)} channels (indices): {discarded_indices}")
    print(f"Discarded channels (names): {discarded_channels}")

    return selected_indices, discarded_indices

if __name__ == "__main__":
    # Test trên participant 1, target arousal
    try:
        print("="*60)
        print("Starting MRMR Channel Selection Test")
        print("="*60)

        selected, discarded = test_mrmr_on_participant(participant_id=1, target="arousal", k=20)

        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        print("Hãy đảm bảo đường dẫn đến dữ liệu DEAP đúng.")