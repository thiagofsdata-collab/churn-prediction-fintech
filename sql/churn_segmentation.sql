-- ============================================================
-- Churn Segmentation Analysis
-- Purpose: Identify which customer segments have the highest
--          churn rates to prioritize retention efforts
-- ============================================================

-- 1. Churn rate by account type
SELECT
    account_type,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS churn_rate_pct
FROM churn_data
GROUP BY account_type
ORDER BY churn_rate_pct DESC;


-- 2. Churn rate by payment method
SELECT
    payment_method,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS churn_rate_pct
FROM churn_data
GROUP BY payment_method
ORDER BY churn_rate_pct DESC;


-- 3. Churn rate by primary product
SELECT
    primary_product,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS churn_rate_pct
FROM churn_data
GROUP BY primary_product
ORDER BY churn_rate_pct DESC;


-- 4. High-risk segment: month-to-month + electronic check
SELECT
    account_type,
    payment_method,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    )                                                   AS churn_rate_pct
FROM churn_data
GROUP BY account_type, payment_method
ORDER BY churn_rate_pct DESC
LIMIT 10;