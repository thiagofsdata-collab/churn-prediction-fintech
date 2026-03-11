"""
evaluate.py
-----------
Standardized evaluation functions for classification models.

Why these metrics:
- ROC-AUC: threshold-independent, measures ranking quality
- F1: balances precision and recall — critical for imbalanced classes
- Precision-Recall AUC: better than ROC-AUC when classes are imbalanced
- Confusion matrix: gives business-interpretable error breakdown
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Full evaluation suite for a binary classification model.
    Returns a dict of metrics for easy comparison across models.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*50}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k.upper():<15} {v}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn', 'Churn'])}")

    return metrics


def plot_evaluation(model, X_test, y_test,
                    model_name: str = "Model",
                    save_path: str = None):
    """
    Plots ROC curve, Precision-Recall curve, and Confusion Matrix.
    Saves to save_path if provided.
    """
    y_pred  = model.predict(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    axes[0].set_title(f"{model_name} — ROC Curve", fontweight="bold")
    axes[0].plot([0,1],[0,1],"k--", linewidth=1)

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title(f"{model_name} — Precision-Recall Curve", fontweight="bold")

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[2],
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    axes[2].set_title(f"{model_name} — Confusion Matrix", fontweight="bold")
    axes[2].set_ylabel("Actual")
    axes[2].set_xlabel("Predicted")

    plt.suptitle(f"{model_name} — Full Evaluation", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Plot saved to {save_path}")

    plt.show()