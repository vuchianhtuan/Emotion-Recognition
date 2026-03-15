"""
models.py
---------
Định nghĩa kiến trúc mô hình phân loại cảm xúc từ EEG:

  1. EEGConvNet  – 1-D CNN trên trục kênh
  2. EEGLSTM     – Bidirectional LSTM
  3. EEGTransformer – Lightweight Transformer Encoder
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#  1. 1-D CNN
# ------------------------------------------------------------------ #
class EEGConvNet(nn.Module):
    """
    Input : (batch, channels=32, n_bands=4)
    Output: (batch, num_classes)
    """

    def __init__(self, n_channels: int = 32, n_bands: int = 4,
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32, 4)
        out = self.conv(x).squeeze(-1)      # (B, 128)
        return self.classifier(out)


# ------------------------------------------------------------------ #
#  2. Bidirectional LSTM
# ------------------------------------------------------------------ #
class EEGLSTM(nn.Module):
    """
    Coi mỗi dải tần là 1 time-step, mỗi kênh là feature dim.
    Input : (batch, n_bands=4, n_channels=32)
    Output: (batch, num_classes)
    """

    def __init__(self, input_size: int = 32, hidden_size: int = 64,
                 num_layers: int = 2, num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x raw: (B, 32, 4) → transpose → (B, 4, 32)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)           # (B, 4, hidden*2)
        out = out[:, -1, :]             # last time-step
        return self.classifier(out)


# ------------------------------------------------------------------ #
#  3. Transformer Encoder
# ------------------------------------------------------------------ #
class EEGTransformer(nn.Module):
    """
    Input : (batch, channels=32, n_bands=4)
    Output: (batch, num_classes)

    Mỗi kênh EEG là 1 token với embedding dimension = n_bands.
    """

    def __init__(self, n_channels: int = 32, n_bands: int = 4,
                 n_heads: int = 4, n_layers: int = 2,
                 dim_feedforward: int = 128, num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_bands, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(64),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32, 4)
        x = self.input_proj(x)          # (B, 32, 64)
        x = self.encoder(x)             # (B, 32, 64)
        x = x.mean(dim=1)               # global avg pooling trên token axis
        return self.classifier(x)


# ------------------------------------------------------------------ #
#  Factory
# ------------------------------------------------------------------ #
MODEL_REGISTRY = {
    "cnn":         EEGConvNet,
    "lstm":        EEGLSTM,
    "transformer": EEGTransformer,
}


def build_model(architecture: str = "lstm", **kwargs) -> nn.Module:
    """Tạo mô hình theo tên. Truyền thêm kwargs cho constructor."""
    if architecture not in MODEL_REGISTRY:
        raise ValueError(f"Kiến trúc không hợp lệ: {architecture}. "
                         f"Chọn trong: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[architecture](**kwargs)
