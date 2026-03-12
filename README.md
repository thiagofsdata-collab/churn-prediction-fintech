# Churn Prediction for Fintech Customers

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green)
![AWS S3](https://img.shields.io/badge/AWS-S3-yellow)
![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey)

End-to-end machine learning project predicting customer churn for a
fintech company. Covers the full ML lifecycle: business understanding,
SQL analysis, feature engineering, model training, explainability,
and cloud artifact storage.

---

## Business Problem

Customer churn is one of the highest-cost problems in fintech.
Acquiring a new customer costs 5-7x more than retaining an existing one.
This project builds a production-minded churn prediction system that:

- Identifies customers at risk **before** they leave
- Quantifies which factors drive churn using SHAP
- Provides actionable retention recommendations backed by model evidence

---

## Project Architecture
```
Raw Data → EDA → SQL Analysis → Feature Engineering
       → Preprocessing Pipeline → Baseline Model → XGBoost
       → SHAP Explainability → AWS S3 Artifact Storage
```

---

## Results

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.8358 | 0.6259 | 0.5058 | 0.8209 |
| XGBoost (Tuned) | **0.8463** | 0.6193 | 0.5033 | 0.8048 |

XGBoost achieves ROC-AUC 0.846. The modest improvement over the baseline
reflects the dataset size (~7k rows) and the quality of feature engineering,
which reduced the gap between linear and ensemble methods.

**Recall of 0.80** means the model catches 8 in 10 churners —
critical for retention campaign targeting.

---

## Top Churn Drivers (SHAP)

| Rank | Feature | Impact |
|---|---|---|
| 1 | Contract type (account_type) | Strongest signal — month-to-month customers churn at 3x the rate |
| 2 | Tenure (months_as_customer) | Short tenure = highest risk window |
| 3 | Monthly transaction volume | High spend without commitment signals price sensitivity |
| 4 | Revenue per month (engineered) | Spending intensity relative to tenure |
| 5 | Primary product | Product type drives significant retention differences |

---

## Project Structure
```
churn-prediction-fintech/
├── data/
│   ├── raw/                  # Original unmodified data
│   └── processed/            # Engineered features, SHAP output
├── notebooks/                # Numbered analysis notebooks
│   ├── 01_eda.ipynb
│   ├── 02_sql_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_baseline_model.ipynb
│   ├── 05_xgboost_model.ipynb
│   ├── 06_shap_explainability.ipynb
│   └── 07_s3_model_registry.ipynb
├── src/                      # Production modules
│   ├── feature_engineering.py
│   ├── preprocessor.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── s3_utils.py
├── sql/                      # SQL analysis scripts
│   ├── churn_segmentation.sql
│   ├── cohort_analysis.sql
│   └── feature_aggregations.sql
├── reports/
│   ├── figures/              # All visualizations (15 plots)
│   └── model_card.md         # Model documentation
└── tests/
    └── test_preprocessor.py  # Unit tests
```

---

## Quickstart
```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-fintech.git
cd churn-prediction-fintech
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

Run tests:
```bash
pytest tests/ -v
```

---

## Key Technical Decisions

**Why XGBoost?**
Handles mixed feature types natively, built-in imbalance control via
`scale_pos_weight`, and first-class SHAP integration for explainability.

**Why Optuna over GridSearchCV?**
Bayesian optimization explores the hyperparameter space more efficiently
than exhaustive grid search — especially important with 9 parameters.

**Why DuckDB for SQL?**
Runs SQL directly on CSV files with zero infrastructure. Used by real
data teams for fast analytical queries without a database server.

**Why `src/` modules instead of notebook code?**
Modular code is testable, reusable, and importable. Notebook-only code
cannot be unit tested or deployed.

**Class imbalance handling:**
`scale_pos_weight` in XGBoost and `class_weight='balanced'` in Logistic
Regression. Evaluated with ROC-AUC and F1 — not raw accuracy.

---

## SQL Analysis

Three analytical SQL scripts in `sql/` executed via DuckDB:

- `churn_segmentation.sql` — churn rates by contract, payment, product
- `cohort_analysis.sql` — lifecycle stage analysis and revenue at risk
- `feature_aggregations.sql` — product adoption scoring and revenue ratios

---

## AWS S3 Integration

Trained model artifacts stored in S3:
```
s3://churn-prediction-fintech-artifacts/
├── models/xgboost_tuned.joblib
└── artifacts/shap_feature_importance.csv
```

See `src/s3_utils.py` for upload, download, and round-trip verification.

---

## Installation — Reproducibility
```bash
# Standard install
pip install -r requirements.txt

# Exact environment reproduction
pip install -r requirements-lock.txt
```

---

## Author

**Thiago** — Data Analyst | Data Scientist
[GitHub](https://github.com/thiagofsdata-collab) •
[LinkedIn](https://linkedin.com/in/thiago-feliciano-40a1461b9/)