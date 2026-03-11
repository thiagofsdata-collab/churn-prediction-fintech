import pandas as pd
import numpy as np


def fix_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known data quality issues in the raw renamed dataset.
    - Converts total_transaction_volume from object to numeric
    - Imputes missing total_transaction_volume for new customers
    """
    df = df.copy()

    df["total_transaction_volume"] = pd.to_numeric(
        df["total_transaction_volume"], errors="coerce"
    )

    # New customers (tenure=0) have no total charges — impute with monthly
    mask = df["total_transaction_volume"].isnull()
    df.loc[mask, "total_transaction_volume"] = df.loc[mask, "monthly_transaction_volume"]

    return df


def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Yes/No columns to integer binary flags (1/0).
    Also handles 'No phone service' and 'No internet service'
    as equivalent to 'No'.
    """
    df = df.copy()

    binary_cols = [
        "has_partner", "has_dependents", "has_phone_service",
        "has_security_product", "has_backup_product",
        "has_protection_plan", "has_support_plan",
        "has_streaming_tv", "has_streaming_movies",
        "is_paperless", "is_senior"
    ]

    no_equivalents = ["No phone service", "No internet service"]

    for col in binary_cols:
        if col in df.columns:
            if col == "is_senior":
                # Already numeric (0/1) in the original dataset
                df[col] = df[col].astype(int)
            else:
                df[col] = df[col].replace(no_equivalents, "No")
                df[col] = (df[col] == "Yes").astype(int)

    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the target variable 'churned' as binary integer.
    Yes -> 1, No -> 0
    """
    df = df.copy()
    df["churned"] = (df["churned"] == "Yes").astype(int)
    return df


def add_revenue_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create revenue-based features that capture spending patterns.
    These are among the strongest predictors in fintech churn models.
    """
    df = df.copy()

    # Low ratio = customer ramping up; High ratio = stable or declining
    df["revenue_per_month"] = (
        df["total_transaction_volume"] /
        df["months_as_customer"].replace(0, 1)  # avoid division by zero
    ).round(2)

    # Values near 1.0 = very consistent; far from 1.0 = irregular
    df["charge_consistency"] = (
        df["monthly_transaction_volume"] /
        df["revenue_per_month"].replace(0, np.nan)
    ).round(4)

    threshold = df["monthly_transaction_volume"].quantile(0.75)
    df["is_high_value"] = (df["monthly_transaction_volume"] > threshold).astype(int)

    return df


def add_tenure_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create tenure-based features. Tenure has a non-linear relationship
    with churn — confirmed in SQL cohort analysis.
    Binning captures this non-linearity explicitly.
    """
    df = df.copy()

    # Early stage flag — first 12 months is highest risk window
    df["is_early_stage"] = (df["months_as_customer"] <= 12).astype(int)

    # Tenure group — aligns with SQL cohort analysis
    df["tenure_group"] = pd.cut(
        df["months_as_customer"],
        bins=[-1, 12, 24, 48, 999],
        labels=["early", "growing", "established", "loyal"],
        right=True
    ).astype(str)

    return df


def add_product_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create product adoption features.
    SQL analysis confirmed: more products = lower churn.
    This captures the 'stickiness' effect.
    """
    df = df.copy()

    product_cols = [
        "has_security_product", "has_backup_product",
        "has_protection_plan", "has_support_plan",
        "has_streaming_tv", "has_streaming_movies"
    ]

    # Sum of all binary product flags (0-6 range)
    df["product_adoption_score"] = df[product_cols].sum(axis=1)

    # High adoption flag (3+ products)
    df["is_multi_product"] = (df["product_adoption_score"] >= 3).astype(int)

    return df


def add_payment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create payment behavior features.
    Autopay customers churn significantly less — confirmed in EDA.
    """
    df = df.copy()

    df["is_autopay"] = df["payment_method"].str.contains(
        "automatic", case=False, na=False
    ).astype(int)

    return df


def run_feature_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master pipeline — applies all feature engineering steps in order.
    This is the single entry point for transforming raw renamed data
    into a model-ready feature set.

    Order matters:
    1. Fix quality issues first
    2. Encode binary flags (used in product features)
    3. Add derived features
    4. Encode target last
    """
    df = fix_data_quality(df)
    df = encode_binary_flags(df)
    df = add_revenue_features(df)
    df = add_tenure_features(df)
    df = add_product_features(df)
    df = add_payment_features(df)
    df = encode_target(df)

    return df