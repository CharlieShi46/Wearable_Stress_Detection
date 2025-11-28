"""
explain.py

SHAP explainability module for ML models.

Supports:
- XGBoost / LightGBM (TreeExplainer)
- Logistic Regression / RF (KernelExplainer fallback)

Outputs:
- Global SHAP importance
- SHAP summary plot
- SHAP force plot for one sample

Run:
    python -m src.explain --config configs/wesad_stress.yaml
"""

import numpy as np
import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from src.config import load_config
from src.data_loading import load_dataset
from src.preprocessing import preprocess_dataset
from src.windows import create_windows
from src.features import extract_features_from_window


# ===========================================================
#  Extract features for entire dataset
# ===========================================================
def extract_all_features(processed, cfg):
    sampling_rate = cfg["preprocessing"]["sampling_rate"]
    channels = [k for k in list(processed.values())[0].keys() if k != "label"]

    # Window segmentation
    windows = create_windows(
        processed,
        sampling_rate,
        cfg["windowing"]["window_size_sec"],
        cfg["windowing"]["stride_sec"]
    )

    X = windows["X"]
    y = windows["y"]

    # Extract feature table
    feat_list = []
    for i in range(len(X)):
        feats = extract_features_from_window(X[i], channels, sampling_rate)
        feat_list.append(feats)

    df = pd.DataFrame(feat_list)
    df["label"] = y
    df = df.dropna()

    return df


# ===========================================================
#  Build SHAP explainer
# ===========================================================
def get_shap_explainer(model, X_sample):
    try:
        # Tree models: XGBoost / LightGBM
        return shap.TreeExplainer(model)
    except:
        # Fallback: model-agnostic
        return shap.KernelExplainer(model.predict_proba, X_sample)


# ===========================================================
#  Main explain function
# ===========================================================
def explain(config_path):

    cfg = load_config(config_path)

    # -------------------------------------------------------
    # 1. Load and preprocess dataset
    # -------------------------------------------------------
    raw = load_dataset(
        cfg["dataset"]["name"],
        cfg["dataset"]["raw_path"]
    )

    processed = preprocess_dataset(
        raw,
        cfg["preprocessing"],
        cfg["dataset"]["interim_path"]
    )

    # -------------------------------------------------------
    # 2. Extract features for SHAP
    # -------------------------------------------------------
    df = extract_all_features(processed, cfg)

    X = df.drop("label", axis=1).values
    y = df["label"].values
    feature_names = df.drop("label", axis=1).columns.tolist()

    # Use small subset for KernelExplainer efficiency
    X_sample = shap.sample(X, 200)

    # -------------------------------------------------------
    # 3. Train ML model
    # -------------------------------------------------------
    from src.models_ml import build_ml_model
    model = build_ml_model(cfg)
    model.fit(X, y)

    # Save model
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, "models/model_shap.pkl")

    # -------------------------------------------------------
    # 4. SHAP Explainer
    # -------------------------------------------------------
    explainer = get_shap_explainer(model, X_sample)

    print("[INFO] Computing SHAP values...")
    shap_values = explainer.shap_values(X_sample)

    # -------------------------------------------------------
    # 5. Summary Plot (global importance)
    # -------------------------------------------------------
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names)

    # -------------------------------------------------------
    # 6. Bar Plot (global feature importance)
    # -------------------------------------------------------
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, plot_type="bar")

    # -------------------------------------------------------
    # 7. Example: per-instance force plot
    # -------------------------------------------------------
    shap.initjs()
    idx = 0  # pick first sample
    shap.force_plot(
        explainer.expected_value,
        shap_values[idx, :],
        X_sample[idx, :],
        feature_names=feature_names
    )


# Entry point
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    explain(args.config)