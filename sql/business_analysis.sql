
-- 1. Customer Segment Performance
-- Analyses order volume, sales, profit and discount by customer segment.

CREATE OR REPLACE VIEW public.vw_customer_segment_performance AS
SELECT
    customer_segment,
    COUNT(*) AS total_orders,
    SUM(total_sales) AS total_sales,
    SUM(order_profit) AS total_profit,
    ROUND(AVG(order_profit)::numeric, 2) AS average_profit_per_order,
    ROUND(AVG(average_discount_rate)::numeric, 4) AS average_discount_rate
FROM public.orders
WHERE is_cancelled = 0
GROUP BY customer_segment;


-- 2. Delivery Performance by Market
-- Compares late delivery rates across markets.

CREATE OR REPLACE VIEW public.vw_delivery_by_market AS
SELECT
    market,
    COUNT(*) AS total_orders,
    SUM(late_delivery_risk) AS late_orders,
    ROUND(
        SUM(late_delivery_risk)::numeric
        / NULLIF(COUNT(*), 0) * 100,
        2
    ) AS late_delivery_percentage
FROM public.orders
WHERE is_cancelled = 0
GROUP BY market;


-- 3. Delivery Performance by Region
-- Identifies regions with higher late delivery rates.

CREATE OR REPLACE VIEW public.vw_delivery_by_region AS
SELECT
    order_region,
    COUNT(*) AS total_orders,
    SUM(late_delivery_risk) AS late_orders,
    ROUND(
        SUM(late_delivery_risk)::numeric
        / NULLIF(COUNT(*), 0) * 100,
        2
    ) AS late_delivery_percentage
FROM public.orders
WHERE is_cancelled = 0
GROUP BY order_region
ORDER BY late_delivery_percentage DESC;


-- 4. Delivery Performance by Shipping Mode

CREATE OR REPLACE VIEW public.vw_delivery_by_shipping_mode AS
SELECT
    shipping_mode,
    COUNT(*) AS total_orders,
    SUM(late_delivery_risk) AS late_orders,
    ROUND(AVG(scheduled_days)::numeric, 2) AS average_scheduled_days,
    ROUND(AVG(actual_shipping_days)::numeric, 2) AS average_actual_shipping_days,
    ROUND(AVG(delay_days)::numeric, 2) AS average_delay_days,
    ROUND(
        SUM(late_delivery_risk)::numeric
        / NULLIF(COUNT(*), 0) * 100,
        2
    ) AS late_delivery_percentage
FROM public.orders
WHERE is_cancelled = 0
GROUP BY shipping_mode;


-- 5. Executive KPIs

CREATE OR REPLACE VIEW public.vw_executive_kpis AS
SELECT
    COUNT(*) AS total_orders,
    SUM(is_cancelled) AS cancelled_orders,
    COUNT(*) FILTER (WHERE is_cancelled = 0) AS non_cancelled_orders,
    ROUND(
        SUM(is_cancelled)::numeric
        / NULLIF(COUNT(*), 0) * 100,
        2
    ) AS cancellation_percentage,
    ROUND(
        AVG(late_delivery_risk) FILTER (WHERE is_cancelled = 0)::numeric * 100,
        2
    ) AS late_delivery_percentage,
    ROUND(
        AVG(delay_days) FILTER (WHERE is_cancelled = 0)::numeric,
        2
    ) AS average_delay_days,
    ROUND(SUM(total_sales) FILTER (WHERE is_cancelled = 0)::numeric, 2) AS total_sales,
    ROUND(SUM(order_profit) FILTER (WHERE is_cancelled = 0)::numeric, 2) AS total_profit
FROM public.orders;


-- 6. Financial Performance by Market

CREATE OR REPLACE VIEW public.vw_financial_by_market AS
SELECT
    market,
    COUNT(*) AS total_orders,
    ROUND(SUM(total_sales)::numeric, 2) AS total_sales,
    ROUND(SUM(total_discount)::numeric, 2) AS total_discount,
    ROUND(SUM(order_profit)::numeric, 2) AS total_profit,
    ROUND(AVG(order_profit)::numeric, 2) AS average_profit_per_order
FROM public.orders
WHERE is_cancelled = 0
GROUP BY market;


-- 7. Monthly Delivery Performance

CREATE OR REPLACE VIEW public.vw_monthly_delivery_performance AS
SELECT
    order_year,
    order_month,
    order_month_name,
    COUNT(*) AS total_orders,
    SUM(late_delivery_risk) AS late_orders,
    ROUND(
        AVG(late_delivery_risk)::numeric * 100,
        2
    ) AS late_delivery_percentage,
    ROUND(AVG(delay_days)::numeric, 2) AS average_delay_days
FROM public.orders
WHERE is_cancelled = 0
GROUP BY order_year, order_month, order_month_name;

