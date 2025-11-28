"""
train.py

Main training pipeline:
1. Load config
2. Load raw dataset
3. Preprocess signals
4. Sliding-window segmentation
5. Feature extraction (if ML)
6. Train ML or DL model
7. Evaluate on validation/test
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.data_loading import load_dataset
from src.preprocessing import preprocess_dataset
from src.windows import create_windows
from src.features import extract_features_from_window
from src.models_ml import build_ml_model
from src.models_dl import build_dl_model
from src.evaluate import evaluate_ml, evaluate_dl


# ===========================================================
#  Extract features for ML models
# ===========================================================
def extract_features_all(X, channels, sampling_rate):
    feature_list = []

    for i in tqdm(range(len(X)), desc="Extracting features"):
        window = X[i]
        feats = extract_features_from_window(window, channels, sampling_rate)
        feature_list.append(feats)

    df = pd.DataFrame(feature_list)
    return df


# ===========================================================
#   DL training helper
# ===========================================================
def train_dl(model, train_loader, val_loader, device, epochs=10):
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()

            optim.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optim.step()

        # Optional validation
        if val_loader:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device).float()
                    yb = yb.to(device).long()
                    out = model(xb)
                    preds = torch.argmax(out, dim=1)
                    correct += (preds == yb).sum().item()
                    total += len(yb)
            acc = correct / total
            print(f"[Epoch {ep+1}] Val Acc: {acc:.4f}")

    return model


# ===========================================================
#   MAIN
# ===========================================================
def main(config_path):

    cfg = load_config(config_path)

    # -------------------------------------------------------
    # 1. Load raw dataset
    # -------------------------------------------------------
    raw = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["raw_path"]
    )

    # -------------------------------------------------------
    # 2. Preprocess signals
    # -------------------------------------------------------
    processed = preprocess_dataset(
        raw,
        cfg["preprocessing"],
        cfg["dataset"]["interim_path"]
    )

    # -------------------------------------------------------
    # 3. Sliding windows
    # -------------------------------------------------------
    windows = create_windows(
        processed,
        cfg["preprocessing"]["sampling_rate"],
        cfg["windowing"]["window_size_sec"],
        cfg["windowing"]["stride_sec"]
    )
    X = windows["X"]  # shape: (N, C, T)
    y = windows["y"]
    subject_ids = windows["subject_ids"]

    channels = [k for k in processed[list(processed.keys())[0]].keys() if k != "label"]
    sampling_rate = cfg["preprocessing"]["sampling_rate"]

    # Decide ML or DL
    model_type = cfg["model"]["type"]

    # -------------------------------------------------------
    # 4A. ML pipeline (extract features)
    # -------------------------------------------------------
    if model_type in ["logistic", "rf", "xgboost"]:
        # Extract features
        features_df = extract_features_all(X, channels, sampling_rate)
        features_df["label"] = y
        df = features_df.dropna()

        X_feat = df.drop("label", axis=1).values
        y_feat = df["label"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X_feat, y_feat, test_size=0.2, random_state=42, stratify=y_feat
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model = build_ml_model(cfg)
        model.fit(X_train, y_train)

        print("[INFO] Evaluating ML model...")
        evaluate_ml(model, X_test, y_test)

    # -------------------------------------------------------
    # 4B. DL pipeline (raw windows)
    # -------------------------------------------------------
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Train/val/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Convert to torch tensors
        X_train_t = torch.tensor(X_train).float()
        y_train_t = torch.tensor(y_train).long()
        X_test_t = torch.tensor(X_test).float()
        y_test_t = torch.tensor(y_test).long()

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=32,
            shuffle=True
        )
        test_loader = DataLoader(
            TensorDataset(X_test_t, y_test_t),
            batch_size=32,
            shuffle=False
        )

        seq_len = X.shape[-1]
        in_channels = X.shape[1]

        model = build_dl_model(cfg, in_channels, seq_len)

        print("[INFO] Training DL model...")
        model = train_dl(model, train_loader, None, device, epochs=8)

        print("[INFO] Evaluating DL model...")
        evaluate_dl(model, test_loader, device)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(args.config)