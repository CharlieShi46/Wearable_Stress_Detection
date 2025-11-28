"""
evaluate.py

Evaluation utilities for:
- ML models (sklearn / xgboost)
- DL models (pytorch)

Provides:
- Accuracy
- F1-score (macro/weighted)
- Classification report
- Confusion matrix
- ROC curve (binary only)
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)
import torch


# ===========================================================
#   Plot confusion matrix
# ===========================================================
def plot_confusion_matrix(cm, classes, title="Confusion Matrix"):
    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()


# ===========================================================
#   ML evaluation (sklearn / xgboost)
# ===========================================================
def evaluate_ml(model, X_test, y_test):
    preds = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Macro:", f1_score(y_test, preds, average="macro"))
    print("F1 Weighted:", f1_score(y_test, preds, average="weighted"))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    classes = sorted(list(set(y_test)))
    plot_confusion_matrix(cm, classes)

    # ROC curve (if binary)
    if len(classes) == 2:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_test)[:, 1]
        else:
            prob = preds

        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], "--")
        plt.title("ROC Curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.show()


# ===========================================================
#   DL evaluation
# ===========================================================
def evaluate_dl(model, test_loader, device="cpu"):
    model.eval()
    preds_list = []
    target_list = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device).float()
            yb = yb.to(device).long()

            out = model(xb)
            preds = torch.argmax(out, dim=1)

            preds_list.extend(preds.cpu().numpy())
            target_list.extend(yb.cpu().numpy())

    preds = np.array(preds_list)
    y_test = np.array(target_list)

    print("Accuracy:", accuracy_score(y_test, preds))
    print("F1 Macro:", f1_score(y_test, preds, average="macro"))
    print("F1 Weighted:", f1_score(y_test, preds, average="weighted"))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    # Confusion matrix
    cm = confusion_matrix(y_test, preds)
    classes = sorted(list(set(y_test)))
    plot_confusion_matrix(cm, classes)