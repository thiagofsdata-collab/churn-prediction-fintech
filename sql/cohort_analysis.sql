-- ============================================================
-- Cohort Analysis by Tenure
-- Purpose: Understand churn risk across customer lifecycle
--          stages to identify when intervention is most needed
-- ============================================================

-- 1. Define tenure cohorts and calculate churn rate per cohort
SELECT
    CASE
        WHEN months_as_customer <= 12 THEN '01_Early (0-12m)'
        WHEN months_as_customer <= 24 THEN '02_Growing (13-24m)'
        WHEN months_as_customer <= 48 THEN '03_Established (25-48m)'
        ELSE '04_Loyal (49m+)'
    END AS tenure_cohort,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS churn_rate_pct,
    ROUND(AVG(monthly_transaction_volume), 2) AS avg_monthly_volume,
    ROUND(AVG(total_transaction_volume), 2) AS avg_total_volume
FROM churn_data
GROUP BY tenure_cohort
ORDER BY tenure_cohort;


-- 2. Churn rate by tenure cohort AND account type
SELECT
    CASE
        WHEN months_as_customer <= 12  THEN '01_Early (0-12m)'
        WHEN months_as_customer <= 24  THEN '02_Growing (13-24m)'
        WHEN months_as_customer <= 48  THEN '03_Established (25-48m)'
        ELSE '04_Loyal (49m+)'
    END AS tenure_cohort,
    account_type,
    COUNT(*) AS total_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS churn_rate_pct
FROM churn_data
GROUP BY tenure_cohort, account_type
ORDER BY tenure_cohort, churn_rate_pct DESC;


-- 3. Average revenue at risk per cohort (churned customers only)
SELECT
    CASE
        WHEN months_as_customer <= 12  THEN '01_Early (0-12m)'
        WHEN months_as_customer <= 24  THEN '02_Growing (13-24m)'
        WHEN months_as_customer <= 48  THEN '03_Established (25-48m)'
        ELSE '04_Loyal (49m+)'
    END AS tenure_cohort,
    COUNT(*) AS churned_customers,
    ROUND(AVG(monthly_transaction_volume), 2) AS avg_monthly_volume_lost,
    ROUND(SUM(monthly_transaction_volume), 2) AS total_monthly_revenue_at_risk
FROM churn_data
WHERE churned = 'Yes'
GROUP BY tenure_cohort
ORDER BY total_monthly_revenue_at_risk DESC;