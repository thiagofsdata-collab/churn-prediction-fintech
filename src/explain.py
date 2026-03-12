"""
explain.py
----------
SHAP-based model explainability for the fintech churn prediction model.

Design decisions:
- TreeExplainer is used for XGBoost (fastest, exact values for tree models)
- Explanations computed on preprocessed data (what the model actually sees)
- Both global and local explanations implemented
- All plots saved to reports/figures/ for README embedding
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def get_explainer(pipeline):
    """
    Creates a SHAP TreeExplainer from a fitted sklearn Pipeline.
    Extracts the XGBoost classifier from the pipeline.
    """
    model = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(model)
    return explainer


def get_shap_values(pipeline, X: pd.DataFrame):
    """
    Transforms X through the preprocessor and computes SHAP values.
    Returns shap_values array and transformed feature matrix.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    explainer = get_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)
    return shap_values, X_transformed


def get_feature_names(pipeline, X: pd.DataFrame) -> list:
    """
    Extracts feature names from the ColumnTransformer in order.
    Handles the numerical + categorical + binary ordering.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(columns, "tolist"):
            feature_names.extend(columns.tolist())
        else:
            feature_names.extend(columns)

    return feature_names


def plot_shap_summary_bar(shap_values, feature_names: list,
                           save_path: str = None):
    """
    Global feature importance — mean absolute SHAP values.
    Bar chart: clean, readable, business-friendly.
    """
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": mean_shap
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(importance_df["feature"], importance_df["importance"],
                   color="#3498db", alpha=0.8, edgecolor="white")
    ax.set_title("Global Feature Importance (Mean |SHAP Value|)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Mean |SHAP Value| — Average Impact on Churn Prediction")

    # Highlight top 5
    top5_threshold = importance_df["importance"].nlargest(5).min()
    for bar, val in zip(bars, importance_df["importance"]):
        color = "#e74c3c" if val >= top5_threshold else "#3498db"
        bar.set_color(color)
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_shap_beeswarm(shap_values, X_transformed,
                        feature_names: list,
                        save_path: str = None):
    """
    Beeswarm plot: shows both direction and magnitude of each
    feature's impact. Red = high feature value, Blue = low.
    This is the most information-dense SHAP visualization.
    """
    shap_explanation = shap.Explanation(
        values=shap_values,
        data=X_transformed,
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(12, 8))
    shap.plots.beeswarm(shap_explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm — Feature Impact Distribution",
              fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    plt.show()


def plot_shap_local(pipeline, X: pd.DataFrame,
                     feature_names: list,
                     customer_index: int,
                     label: str = "Customer",
                     save_path: str = None):
    """
    Local explanation for a single customer.
    Waterfall plot shows exactly which features pushed the
    prediction toward or away from churn.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    explainer = get_explainer(pipeline)
    shap_values = explainer.shap_values(X_transformed)

    shap_explanation = shap.Explanation(
        values=shap_values[customer_index],
        base_values=explainer.expected_value,
        data=X_transformed[customer_index],
        feature_names=feature_names
    )

    fig = plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_explanation, max_display=12, show=False)
    plt.title(f"Local Explanation — {label}",
              fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Saved: {save_path}")
    plt.show()