
-- 1. Check total number of records
SELECT COUNT(*) AS total_orders
FROM public.orders;

-- 2. Check duplicate Order IDs
SELECT
    order_id,
    COUNT(*) AS duplicate_count
FROM public.orders
GROUP BY order_id
HAVING COUNT(*) > 1;

-- 3. Check missing Order IDs
SELECT COUNT(*) AS missing_order_ids
FROM public.orders
WHERE order_id IS NULL;

-- 4. Check missing Order Dates
SELECT COUNT(*) AS missing_order_dates
FROM public.orders
WHERE order_date IS NULL;

-- 5. Check missing Delivery Status
SELECT COUNT(*) AS missing_delivery_status
FROM public.orders
WHERE delivery_status IS NULL;

-- 6. Check missing Late Delivery Risk values
SELECT COUNT(*) AS missing_risk_labels
FROM public.orders
WHERE late_delivery_risk IS NULL;

-- 7. Check delivery risk distribution
SELECT
    late_delivery_risk,
    COUNT(*) AS order_count
FROM public.orders
GROUP BY late_delivery_risk
ORDER BY late_delivery_risk;

-- 8. Check cancelled and non-cancelled orders
SELECT
    is_cancelled,
    COUNT(*) AS order_count
FROM public.orders
GROUP BY is_cancelled
ORDER BY is_cancelled;
