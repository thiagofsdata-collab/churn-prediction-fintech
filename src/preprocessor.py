"""
preprocessor.py
---------------
Defines the sklearn preprocessing pipeline for the fintech
churn prediction model.

Design decisions:
- ColumnTransformer handles numerical and categorical features separately
- Pipeline ensures no data leakage (fit only on train, transform on both)
- OrdinalEncoder used for tree-based models (XGBoost handles this better
  than OneHotEncoder for high-cardinality or ordinal features)
- StandardScaler included for Logistic Regression baseline compatibility
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


# ── Feature groups ────────────────────────────────────────────────────────────

NUMERICAL_FEATURES = [
    "months_as_customer",
    "monthly_transaction_volume",
    "total_transaction_volume",
    "revenue_per_month",
    "charge_consistency",
    "product_adoption_score",
]

BINARY_FEATURES = [
    "is_senior",
    "has_partner",
    "has_dependents",
    #"has_phone_service",
    "has_security_product",
    "has_backup_product",
    "has_protection_plan",
    "has_support_plan",
    "has_streaming_tv",
    "has_streaming_movies",
    "is_paperless",
    "is_autopay",
    "is_high_value",
    "is_early_stage",
    "is_multi_product",
]

CATEGORICAL_FEATURES = [
    "gender",
    "primary_product",
    "account_type",
    "payment_method",
    "tenure_group",
]

TARGET = "churned"
DROP_COLS = ["customer_id", "has_multiple_lines", "has_phone_service"]


# ── Sub-pipelines ─────────────────────────────────────────────────────────────

def build_numerical_pipeline() -> Pipeline:
    """
    Numerical features:
    - Impute missing with median (robust to outliers)
    - Scale with StandardScaler (required for Logistic Regression)
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_categorical_pipeline() -> Pipeline:
    """
    Categorical features:
    - Impute missing with most frequent value
    - OrdinalEncoder: works well with tree-based models
      handle_unknown='use_encoded_value' prevents errors on unseen categories
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
    ])


def build_binary_pipeline() -> Pipeline:
    """
    Binary features are already 0/1 integers.
    Impute only — no scaling needed (already bounded).
    """
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])


# ── Master preprocessor ───────────────────────────────────────────────────────

def build_preprocessor() -> ColumnTransformer:
    """
    Combines all sub-pipelines into a single ColumnTransformer.
    remainder='drop' explicitly drops unused columns (customer_id, etc.)
    """
    return ColumnTransformer(
        transformers=[
            ("numerical",    build_numerical_pipeline(),    NUMERICAL_FEATURES),
            ("categorical",  build_categorical_pipeline(),  CATEGORICAL_FEATURES),
            ("binary",       build_binary_pipeline(),       BINARY_FEATURES),
        ],
        remainder="drop"
    )


def get_feature_names() -> list:
    """Returns ordered list of all feature names after transformation."""
    return NUMERICAL_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES


def prepare_X_y(df: pd.DataFrame):
    """
    Split a feature-engineered DataFrame into X and y.
    Drops non-feature columns and separates the target.
    """
    X = df.drop(columns=[TARGET] + DROP_COLS, errors="ignore")
    y = df[TARGET]
    return X, y