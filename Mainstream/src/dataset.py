"""
dataset.py
----------
Custom PyTorch Dataset cho EEG DEAP.

Mỗi sample: (features, label_valence, label_arousal)
  - features : Tensor (channels, n_bands)
  - label    : 0 (Low) hoặc 1 (High) theo ngưỡng 5.0
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import extract_features


LABEL_THRESHOLD = 5.0  # Phân ngưỡng nhị phân Low / High


class DEAPDataset(Dataset):
    """
    Đọc file .dat của DEAP, trích xuất đặc trưng PSD/DE và trả về Tensor.

    Args:
        data_dir  : Đường dẫn đến thư mục chứa các file s01.dat … s32.dat
        target    : "valence" hoặc "arousal"
        mode      : "psd" hoặc "de"
        transform : Callable tuỳ chọn áp dụng sau khi lấy features
    """

    def __init__(self, data_dir: str, target: str = "valence",
                 mode: str = "psd", transform=None):
        self.data_dir = data_dir
        self.target = target
        self.mode = mode
        self.transform = transform

        self.samples: list[np.ndarray] = []
        self.labels: list[int] = []

        self._load_all()

    # ------------------------------------------------------------------ #
    def _load_all(self):
        dat_files = sorted(
            f for f in os.listdir(self.data_dir) if f.endswith(".dat")
        )
        if not dat_files:
            raise FileNotFoundError(f"Không tìm thấy file .dat trong: {self.data_dir}")

        label_idx = 0 if self.target == "valence" else 1  # DEAP: [valence, arousal, ...]

        for fname in dat_files:
            path = os.path.join(self.data_dir, fname)
            with open(path, "rb") as f:
                subject = pickle.load(f, encoding="latin1")

            data   = subject["data"]    # (40 trials, 40 channels, 8064 samples)
            labels = subject["labels"]  # (40 trials, 4 labels)

            for trial_idx in range(data.shape[0]):
                trial     = data[trial_idx, :32, :]          # 32 EEG channels
                raw_label = labels[trial_idx, label_idx]
                binary    = int(raw_label >= LABEL_THRESHOLD)

                feats = extract_features(trial, mode=self.mode)
                self.samples.append(feats)
                self.labels.append(binary)

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x = torch.tensor(self.samples[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx],  dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y
