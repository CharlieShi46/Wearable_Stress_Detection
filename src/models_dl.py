"""
models_dl.py

Two baseline deep learning models:
- 1D CNN
- LSTM

Unified build:
    model = build_dl_model(cfg, input_channels, seq_len)
"""

import torch
import torch.nn as nn


# ===========================================================
# 1D CNN
# ===========================================================
class CNN1D(nn.Module):
    def __init__(self, in_channels, seq_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(64, 2)  # binary output

    def forward(self, x):
        # x: batch × channels × seq_len
        x = self.net(x)
        x = x.squeeze(-1)
        return self.fc(x)


# ===========================================================
# LSTM
# ===========================================================
class LSTMModel(nn.Module):
    def __init__(self, in_channels, seq_len, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: batch × channels × seq_len → batch × seq_len × channels
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# ===========================================================
# Builder
# ===========================================================
def build_dl_model(cfg, in_channels, seq_len):
    model_type = cfg["model"]["type"]

    if model_type == "cnn":
        return CNN1D(in_channels, seq_len)

    if model_type == "lstm":
        return LSTMModel(in_channels, seq_len)

    raise ValueError(f"Unknown DL model: {model_type}")