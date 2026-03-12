# Model Card — Fintech Customer Churn Prediction

## Model Details

| Field | Value |
|---|---|
| Model type | XGBoost Classifier |
| Version | 1.0 |
| Training date | March 2026 |
| Framework | XGBoost 3.2.0 + scikit-learn 1.8.0 |
| Artifact location | s3://churn-prediction-fintech-artifacts/models/ |

## Intended Use

**Primary use case:** Identify fintech customers at high risk of churning
within the next billing cycle to enable proactive retention interventions.

**Intended users:** Retention marketing teams, customer success managers,
data science teams building retention scoring systems.

**Out-of-scope uses:** This model should not be used for credit scoring,
loan decisions, or any regulatory compliance decisions.

## Dataset

| Field | Value |
|---|---|
| Source | IBM Telco Churn dataset (adapted for fintech context) |
| Size | 7,043 customers |
| Features | 26 (original) + 8 engineered |
| Target | Binary churn (0 = retained, 1 = churned) |
| Class balance | 73.5% retained / 26.5% churned |

## Performance

| Metric | Logistic Regression (Baseline) | XGBoost (Tuned) |
|---|---|---|
| ROC-AUC | 0.8358 | 0.8463 |
| F1 Score | 0.6259 | 0.6193 |
| Precision | 0.5058 | 0.5033 |
| Recall | 0.8209 | 0.8048 |

Evaluation performed on a held-out test set (20% of data, stratified split).

## Key Churn Drivers (SHAP)

| Rank | Feature | Mean |SHAP| | Business Interpretation |
|---|---|---|---|
| 1 | account_type | 0.7496 | Contract type is the dominant predictor |
| 2 | months_as_customer | 0.3357 | Short tenure = high churn risk |
| 3 | monthly_transaction_volume | 0.2405 | High spend without commitment = risk |
| 4 | revenue_per_month | 0.1934 | Engineered feature — spending intensity |
| 5 | primary_product | 0.1634 | Product type drives retention differences |

## Ethical Considerations

- Gender is included as a feature but has near-zero SHAP importance (0.0077)
- Model does not use race, nationality, or religion
- Predictions should inform, not automate, retention decisions
- False positives (predicting churn for retained customers) result in
  unnecessary retention spend — not harm to customers

## Limitations

- Dataset represents a single market segment (~7k customers)
- Performance may degrade on customer populations with different
  contract or payment distributions
- Model requires retraining as product offerings evolve

## Recommendations for Production

1. Retrain monthly with fresh customer data
2. Monitor ROC-AUC and recall on live predictions
3. Set decision threshold based on retention campaign ROI,
   not default 0.5 — consider 0.35-0.40 to maximize recall
4. A/B test retention interventions on model-flagged customers