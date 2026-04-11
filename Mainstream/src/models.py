

"""
models.py
---------
Định nghĩa kiến trúc mô hình phân loại cảm xúc từ EEG.

Sử dụng PyTorch để tương thích với training code trong app/main.py.
"""

import torch
import torch.nn as nn


class MRMRLSTM(nn.Module):
    """
    BiLSTM model matching DEAP-Emotion-Recognition architecture.
    
    Args:
        input_size: Number of features per time step
        hidden_size: Hidden size for LSTM layers
        num_layers: Number of LSTM layers
        num_classes: Number of output classes (2 for binary classification)
        dropout: Dropout rate
    """
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=5, num_classes=2, dropout=0.5):
        super(MRMRLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Additional dropout layers
        self.dropout1 = nn.Dropout(0.6)
        self.dropout2 = nn.Dropout(0.6)
        self.dropout3 = nn.Dropout(0.6)
        self.dropout4 = nn.Dropout(0.4)
        
        # Output layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # *2 for bidirectional
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # LSTM layers with manual dropout application
        out, _ = self.lstm(x)  # out: (batch_size, seq_len, hidden_size*2)
        
        # Apply dropout to LSTM outputs at different layers
        # This approximates the original architecture
        out = self.dropout1(out)
        out = self.dropout2(out) 
        out = self.dropout3(out)
        out = self.dropout4(out)
        
        # Take the last time step output
        out = out[:, -1, :]  # (batch_size, hidden_size*2)
        
        # Output layer
        out = self.fc(out)  # (batch_size, num_classes)
        return out


def build_model(arch="mrmr_lstm", seq_len=None, dropout=0.5, input_size=1):
    """
    Build PyTorch model matching DEAP-Emotion-Recognition architecture.

    Args:
        arch: Model architecture (only 'mrmr_lstm' supported)
        seq_len: Sequence length (not used, kept for compatibility)
        dropout: Dropout rate
        input_size: Number of features per time step

    Returns:
        PyTorch nn.Module
    """
    if arch != "mrmr_lstm":
        raise ValueError("Only 'mrmr_lstm' architecture supported for DEAP compatibility")
    
    model = MRMRLSTM(
        input_size=input_size,
        hidden_size=128,
        num_layers=5,
        num_classes=2,
        dropout=dropout
    )
    
    return model
    model.add(Dropout(0.4))

    model.add(Dense(units=16))
    model.add(Activation('relu'))

    model.add(Dense(units=2))
    model.add(Activation("softmax"))

    model.compile(optimizer="adam", loss=categorical_crossentropy, metrics=["accuracy"])
    print(model.summary())
    return model


def training(y_train, y_test, x_train, x_test, epochs, batch_size=256):
    """
    Train model matching DEAP-Emotion-Recognition training function.

    Args:
        y_train, y_test: One-hot encoded labels
        x_train, x_test: Input features
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        Training history and trained model
    """
    model = build_model(input_shape=(x_train.shape[1], 1))

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(x_test, y_test)
    )

    return history, model
