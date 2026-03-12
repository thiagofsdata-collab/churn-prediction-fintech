"""
test_preprocessor.py
--------------------
Unit tests for the preprocessing pipeline.
Tests that the pipeline transforms data correctly
and produces expected output shapes and types.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from feature_engineering import run_feature_engineering_pipeline
from preprocessor import build_preprocessor, prepare_X_y


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe mimicking fintech churn dataset."""
    return pd.DataFrame({
        "customer_id":                ["C001", "C002", "C003"],
        "gender":                     ["Male", "Female", "Male"],
        "is_senior":                  [0, 1, 0],
        "has_partner":                ["Yes", "No", "Yes"],
        "has_dependents":             ["No", "No", "Yes"],
        "months_as_customer":         [12, 0, 48],
        "has_phone_service":          ["Yes", "Yes", "No"],
        "has_multiple_lines":         ["No", "Yes", "No phone service"],
        "primary_product":            ["DSL", "Fiber optic", "No"],
        "has_security_product":       ["No", "Yes", "No"],
        "has_backup_product":         ["Yes", "No", "No"],
        "has_protection_plan":        ["No", "No", "Yes"],
        "has_support_plan":           ["Yes", "No", "No"],
        "has_streaming_tv":           ["No", "Yes", "No"],
        "has_streaming_movies":       ["No", "No", "Yes"],
        "account_type":               ["Month-to-month", "One year", "Two year"],
        "is_paperless":               ["Yes", "No", "Yes"],
        "payment_method":             ["Electronic check", "Bank transfer (automatic)", "Credit card (automatic)"],
        "monthly_transaction_volume": [70.5, 45.0, 90.0],
        "total_transaction_volume":   ["846.0", " ", "4320.0"],
        "churned":                    ["Yes", "No", "No"],
    })


def test_feature_engineering_runs(sample_df):
    """Pipeline should run without errors on valid input."""
    result = run_feature_engineering_pipeline(sample_df)
    assert result is not None
    assert isinstance(result, pd.DataFrame)


def test_feature_engineering_adds_columns(sample_df):
    """Engineering pipeline should add new columns."""
    result = run_feature_engineering_pipeline(sample_df)
    expected_new_cols = [
        "revenue_per_month", "charge_consistency", "is_high_value",
        "is_early_stage", "tenure_group", "product_adoption_score",
        "is_multi_product", "is_autopay"
    ]
    for col in expected_new_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_no_nulls_after_engineering(sample_df):
    """No null values should remain after feature engineering."""
    result = run_feature_engineering_pipeline(sample_df)
    null_counts = result.isnull().sum()
    assert null_counts.sum() == 0, f"Null values found:\n{null_counts[null_counts > 0]}"


def test_target_is_binary(sample_df):
    """Target variable should be encoded as 0 and 1 only."""
    result = run_feature_engineering_pipeline(sample_df)
    assert set(result["churned"].unique()).issubset({0, 1})


def test_preprocessor_output_shape(sample_df):
    """Preprocessor should return correct number of features."""
    df = run_feature_engineering_pipeline(sample_df)
    X, y = prepare_X_y(df)
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == 3
    assert X_transformed.shape[1] > 0


def test_prepare_X_y_drops_target(sample_df):
    """prepare_X_y should remove target from feature matrix."""
    df = run_feature_engineering_pipeline(sample_df)
    X, y = prepare_X_y(df)
    assert "churned" not in X.columns
    assert len(y) == len(X)