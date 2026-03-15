"""
utils.py
--------
Các hàm bổ trợ: seed, checkpoint I/O, visualization.
"""

import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (phù hợp server/Colab)
import matplotlib.pyplot as plt


# ------------------------------------------------------------------ #
#  Reproducibility
# ------------------------------------------------------------------ #
def set_seed(seed: int = 42):
    """Cố định seed để kết quả có thể tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------ #
#  Checkpoint
# ------------------------------------------------------------------ #
def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "epoch":      epoch,
        "model":      model.state_dict(),
        "optimizer":  optimizer.state_dict(),
    }, path)


def load_checkpoint(model: torch.nn.Module, path: str,
                    optimizer: torch.optim.Optimizer | None = None) -> int:
    """Load checkpoint vào model (và optimizer nếu cần). Trả về epoch đã lưu."""
    ckpt  = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0)


# ------------------------------------------------------------------ #
#  Visualization
# ------------------------------------------------------------------ #
def plot_history(history: dict, save_dir: str = "reports/figures",
                 title: str = "Training History"):
    """
    Vẽ Loss và Accuracy theo epoch và lưu vào save_dir.

    Args:
        history : dict với keys train_loss, val_loss, train_acc, val_acc
        save_dir: thư mục lưu ảnh
        title   : tiêu đề biểu đồ (cũng dùng làm tên file)
    """
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title)

    # Loss
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"],   label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"],   label="Val Acc")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    safe_title = title.replace(" ", "_").replace("–", "-")
    out_path   = os.path.join(save_dir, f"{safe_title}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved → {out_path}")
