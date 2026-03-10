-- ============================================================
-- Feature Aggregations
-- Purpose: Generate SQL-derived features that will feed into
--          the ML pipeline. Shows integration between SQL
--          analysis and feature engineering.
-- ============================================================

-- 1. Revenue intensity ratio
--    High monthly charges relative to tenure = potential churn risk
SELECT
    customer_id,
    months_as_customer,
    monthly_transaction_volume,
    total_transaction_volume,
    ROUND(
        monthly_transaction_volume / NULLIF(months_as_customer, 0), 2
    ) AS revenue_per_month_ratio,
    churned
FROM churn_data
ORDER BY revenue_per_month_ratio DESC
LIMIT 20;


-- 2. Product adoption score
--    Count how many additional products/services a customer uses
SELECT
    customer_id,
    (CASE WHEN has_security_product = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN has_backup_product = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN has_protection_plan = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN has_support_plan = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN has_streaming_tv = 'Yes' THEN 1 ELSE 0 END +
     CASE WHEN has_streaming_movies = 'Yes' THEN 1 ELSE 0 END) AS product_adoption_score,
    churned
FROM churn_data;


-- 3. Churn rate by product adoption score
--    Tests hypothesis: more products = lower churn (stickiness)
SELECT
    product_adoption_score,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        SUM(CASE WHEN churned = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2
    ) AS churn_rate_pct
FROM (
    SELECT
        customer_id,
        (CASE WHEN has_security_product = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN has_backup_product = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN has_protection_plan = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN has_support_plan = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN has_streaming_tv = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN has_streaming_movies = 'Yes' THEN 1 ELSE 0 END) AS product_adoption_score,
        churned
    FROM churn_data
) subq
GROUP BY product_adoption_score
ORDER BY product_adoption_score;