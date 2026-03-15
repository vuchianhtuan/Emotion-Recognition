"""
train.py
--------
Script huấn luyện và lưu mô hình EEG Emotion Recognition.

Cách chạy:
    python -m src.train --target valence --arch lstm --epochs 50
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .dataset import DEAPDataset
from .models  import build_model
from .utils   import set_seed, save_checkpoint, plot_history


# ------------------------------------------------------------------ #
#  Config mặc định
# ------------------------------------------------------------------ #
DEFAULT_DATA_DIR  = "data/raw"
DEFAULT_MODEL_DIR = "models"
BATCH_SIZE        = 64
LR                = 1e-3
WEIGHT_DECAY      = 1e-4
VAL_SPLIT         = 0.2


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
#  Main
# ------------------------------------------------------------------ #
def main(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # Dataset
    dataset = DEAPDataset(args.data_dir, target=args.target, mode=args.feat)
    val_size   = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Emotion Recognition – Train")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--target",   default="valence", choices=["valence", "arousal"])
    parser.add_argument("--arch",     default="lstm",    choices=["cnn", "lstm", "transformer"])
    parser.add_argument("--feat",     default="psd",     choices=["psd", "de"])
    parser.add_argument("--epochs",   default=50, type=int)
    parser.add_argument("--seed",     default=42, type=int)
    main(parser.parse_args())
