"""
train.py
--------
Script huấn luyện và lưu mô hình EEG Emotion Recognition.

Cách chạy:
    # PSD/DE pipeline (PyTorch CNN/LSTM/Transformer)
    python -m src.train --target valence --arch lstm --epochs 50

    # MRMR pipeline (Bidirectional LSTM – khớp kiến trúc gốc DEAP)
    python -m src.train --target arousal --feat mrmr --arch mrmr_lstm --data-dir data/raw --epochs 200
"""

import argparse
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from .dataset          import DEAPDataset
from .models           import build_model
from .utils            import set_seed, save_checkpoint, plot_history
from .mrmr_selection   import (
    preprocess_subject_fft,
    run_mrmr_selection,
    build_mrmr_dataset,
    prepare_for_lstm,
)


# ------------------------------------------------------------------ #
#  Config mặc định
# ------------------------------------------------------------------ #
DEFAULT_DATA_DIR  = "data/raw"
DEFAULT_MODEL_DIR = "models"
BATCH_SIZE        = 256
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
VAL_SPLIT         = 0.2
MRMR_COMPONENTS   = 20   # Top-K channels


# ------------------------------------------------------------------ #
#  Train / Eval một epoch
# ------------------------------------------------------------------ #
def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            total      += len(y)

    return total_loss / total, correct / total


# ------------------------------------------------------------------ #
#  MRMR training pipeline
# ------------------------------------------------------------------ #
def train_mrmr(args):
    """Full MRMR training pipeline.

    Steps:
      1. Load all DEAP .dat files
      2. Per-subject FFT preprocessing (5 bands, sliding window)
      3. Per-subject MRMR channel selection
      4. Build aggregated train / test split
      5. Train EEGMRMRLSTMNet on the MRMR features
    """
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] MRMR pipeline | Target: {args.target} | Device: {device}")

    # ── 1. Load & preprocess ──────────────────────────────────────── #
    dat_files = sorted(f for f in os.listdir(args.data_dir) if f.endswith(".dat"))
    if not dat_files:
        raise FileNotFoundError(f"No .dat files found in {args.data_dir}")

    all_subjects_data: list = []
    all_selected_channels: list = []
    channel_backup = []

    for fname in dat_files:
        path = os.path.join(args.data_dir, fname)
        with open(path, "rb") as f:
            subject = pickle.load(f, encoding="latin1")

        print(f"  Preprocessing {fname} …")
        preprocessed = preprocess_subject_fft(subject)
        all_subjects_data.append(preprocessed)

        print(f"  Running MRMR on {fname} …")
        selected = run_mrmr_selection(
            preprocessed, classify_type=args.target, K=MRMR_COMPONENTS
        )
        all_selected_channels.append(selected)
        channel_backup.append({"file": fname, "channels": selected})
        print(f"    Selected channels: {selected}")

    # ── 2. Save channel selection results ────────────────────────── #
    os.makedirs(DEFAULT_MODEL_DIR, exist_ok=True)
    channel_file = os.path.join(
        DEFAULT_MODEL_DIR, f"mrmr_channels_{args.target}.pkl"
    )
    with open(channel_file, "wb") as f:
        pickle.dump(channel_backup, f)
    print(f"[INFO] MRMR channel info saved → {channel_file}")

    # ── 3. Build dataset ─────────────────────────────────────────── #
    x_train_raw, y_train_raw, x_test_raw, y_test_raw = build_mrmr_dataset(
        all_subjects_data, all_selected_channels
    )

    x_train, y_train_bin, x_test, y_test_bin = prepare_for_lstm(
        x_train_raw, x_test_raw, y_train_raw, y_test_raw,
        classify_type=args.target,
    )

    print(f"[INFO] Train: {x_train.shape}, {y_train_bin.shape}")
    print(f"[INFO] Test : {x_test.shape},  {y_test_bin.shape}")

    # ── 4. DataLoaders ────────────────────────────────────────────── #
    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_bin, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test_bin, dtype=torch.long)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False)

    # ── 5. Model ──────────────────────────────────────────────────── #
    seq_len = x_train.shape[1]   # K * N_FREQ
    model = build_model("mrmr_lstm", seq_len=seq_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, test_loader,  criterion, None,      device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train {tr_loss:.4f}/{tr_acc:.3f} | Val {va_loss:.4f}/{va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = os.path.join(
                DEFAULT_MODEL_DIR, f"{args.target}_mrmr_lstm_best.pth"
            )
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}")

    plot_history(history, save_dir="reports/figures",
                 title=f"{args.target.capitalize()} – MRMR LSTM")
    print(f"\n[DONE] Best val acc: {best_val_acc:.4f}")


# ------------------------------------------------------------------ #
#  Standard PSD/DE training pipeline
# ------------------------------------------------------------------ #
def train_standard(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    dataset = DEAPDataset(args.data_dir, target=args.target, mode=args.feat)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(args.arch).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, None,      device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        print(f"Epoch {epoch:03d}/{args.epochs} | "
              f"Train {tr_loss:.4f}/{tr_acc:.3f} | Val {va_loss:.4f}/{va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            ckpt_path = os.path.join(DEFAULT_MODEL_DIR,
                                     f"{args.target}_{args.arch}_v1.pth")
            save_checkpoint(model, optimizer, epoch, ckpt_path)
            print(f"  ✓ Saved checkpoint → {ckpt_path}")

    plot_history(history, save_dir="reports/figures",
                 title=f"{args.target.capitalize()} – {args.arch.upper()}")
    print(f"\n[DONE] Best val acc: {best_val_acc:.4f}")


# ------------------------------------------------------------------ #
def main(args):
    if args.feat == "mrmr":
        train_mrmr(args)
    else:
        train_standard(args)


# ------------------------------------------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Emotion Recognition – Train")
    parser.add_argument("--data-dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--target",     default="valence", choices=["valence", "arousal"])
    parser.add_argument("--arch",       default="lstm",
                        choices=["cnn", "lstm", "transformer", "mrmr_lstm"])
    parser.add_argument("--feat",       default="psd",     choices=["psd", "de", "mrmr"])
    parser.add_argument("--epochs",     default=50, type=int)
    parser.add_argument("--seed",       default=42, type=int)
    parser.add_argument("--batch-size", default=BATCH_SIZE, type=int,
                        help="Mini-batch size (default: %(default)s)")
    main(parser.parse_args())
