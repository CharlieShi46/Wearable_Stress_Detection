"""
models_ml.py

Traditional ML models:
- Logistic Regression
- Random Forest
- XGBoost

Unified interface:
    model = build_ml_model(cfg)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb


def build_ml_model(cfg):
    model_type = cfg["model"]["type"]
    params = cfg["model"].get("params", {})

    if model_type == "logistic":
        return LogisticRegression(max_iter=500, **params)

    elif model_type == "rf":
        return RandomForestClassifier(**params)

    elif model_type == "xgboost":
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            use_label_encoder=False,
            **params
        )

    else:
        raise ValueError(f"Unknown ML model: {model_type}")