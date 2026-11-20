
# ============================================================
# SUPPLY CHAIN RISK INTELLIGENCE
# Phase 3: Exploratory Data Analysis (EDA)
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------------------------------------------------
# 1. PATHS AND DATA LOADING
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

PROCESSED_PATH = PROJECT_DIR / "data" / "processed"
REPORT_PATH = PROJECT_DIR / "reports"
FIGURE_PATH = REPORT_PATH / "figures"

REPORT_PATH.mkdir(parents=True, exist_ok=True)
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

ORDERS_PATH = PROCESSED_PATH / "orders_cleaned.csv"
MODEL_PATH = PROCESSED_PATH / "orders_for_prediction.csv"

orders = pd.read_csv(ORDERS_PATH)
model_data = pd.read_csv(MODEL_PATH)

orders["order_date"] = pd.to_datetime(
    orders["order_date"],
    errors="coerce"
)

print("=" * 60)
print("PHASE 3: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nOrder-level dataset shape:", orders.shape)
print("Prediction dataset shape:", model_data.shape)


# ------------------------------------------------------------
# 2. DELIVERY STATUS ANALYSIS
# ------------------------------------------------------------

print("\n1. DELIVERY STATUS DISTRIBUTION")

delivery_counts = orders["delivery_status"].value_counts()

print(delivery_counts)

delivery_counts.to_csv(
    REPORT_PATH / "eda_delivery_status.csv"
)

plt.figure(figsize=(9, 5))
delivery_counts.plot(kind="bar")

plt.title("Distribution of Delivery Status")
plt.xlabel("Delivery Status")
plt.ylabel("Number of Orders")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "delivery_status_distribution.png"
)
plt.close()


# ------------------------------------------------------------
# 3. LATE DELIVERY RISK ANALYSIS
# ------------------------------------------------------------

print("\n2. LATE DELIVERY RISK DISTRIBUTION")

risk_counts = model_data["late_delivery_risk"].value_counts()

print(risk_counts)

risk_percentages = (
    model_data["late_delivery_risk"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nRisk percentages:")
print(risk_percentages)

risk_counts.to_csv(
    REPORT_PATH / "eda_late_delivery_risk.csv"
)

plt.figure(figsize=(7, 5))
risk_counts.sort_index().plot(kind="bar")

plt.title("Late Delivery Risk Distribution")
plt.xlabel("Late Delivery Risk (0 = No, 1 = Yes)")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "late_delivery_risk_distribution.png"
)
plt.close()


# ------------------------------------------------------------
# 4. DELIVERY DELAY ANALYSIS
# ------------------------------------------------------------

print("\n3. DELIVERY DELAY ANALYSIS")

# Exclude cancelled orders from delay analysis
delivered_orders = orders[
    orders["delivery_status"] != "Shipping canceled"
].copy()

delay_counts = delivered_orders["delay_days"].value_counts().sort_index()

print("\nDelay days distribution:")
print(delay_counts)

delay_summary = delivered_orders["delay_days"].describe()

print("\nDelay summary:")
print(delay_summary)

delay_counts.to_csv(
    REPORT_PATH / "eda_delay_distribution.csv"
)

delay_summary.to_csv(
    REPORT_PATH / "eda_delay_summary.csv"
)

plt.figure(figsize=(9, 5))
delay_counts.plot(kind="bar")

plt.title("Distribution of Delivery Delay Days")
plt.xlabel("Delay Days (Negative = Early, Positive = Late)")
plt.ylabel("Number of Orders")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "delivery_delay_distribution.png"
)
plt.close()


# ------------------------------------------------------------
# 5. SHIPPING MODE ANALYSIS
# ------------------------------------------------------------

print("\n4. LATE DELIVERY RATE BY SHIPPING MODE")

shipping_analysis = (
    model_data.groupby("shipping_mode")
    .agg(
        total_orders=("late_delivery_risk", "count"),
        late_orders=("late_delivery_risk", "sum"),
        late_delivery_rate=("late_delivery_risk", "mean")
    )
    .reset_index()
)

shipping_analysis["late_delivery_rate"] *= 100

shipping_analysis["late_delivery_rate"] = (
    shipping_analysis["late_delivery_rate"].round(2)
)

shipping_analysis = shipping_analysis.sort_values(
    "late_delivery_rate",
    ascending=False
)

print(shipping_analysis)

shipping_analysis.to_csv(
    REPORT_PATH / "eda_shipping_mode_analysis.csv",
    index=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    shipping_analysis["shipping_mode"],
    shipping_analysis["late_delivery_rate"]
)

plt.title("Late Delivery Rate by Shipping Mode")
plt.xlabel("Shipping Mode")
plt.ylabel("Late Delivery Rate (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "late_rate_by_shipping_mode.png"
)
plt.close()


# ------------------------------------------------------------
# 6. MARKET ANALYSIS
# ------------------------------------------------------------

print("\n5. LATE DELIVERY RATE BY MARKET")

market_analysis = (
    model_data.groupby("market")
    .agg(
        total_orders=("late_delivery_risk", "count"),
        late_orders=("late_delivery_risk", "sum"),
        late_delivery_rate=("late_delivery_risk", "mean")
    )
    .reset_index()
)

market_analysis["late_delivery_rate"] *= 100

market_analysis["late_delivery_rate"] = (
    market_analysis["late_delivery_rate"].round(2)
)

market_analysis = market_analysis.sort_values(
    "late_delivery_rate",
    ascending=False
)

print(market_analysis)

market_analysis.to_csv(
    REPORT_PATH / "eda_market_analysis.csv",
    index=False
)

plt.figure(figsize=(9, 5))

plt.bar(
    market_analysis["market"],
    market_analysis["late_delivery_rate"]
)

plt.title("Late Delivery Rate by Market")
plt.xlabel("Market")
plt.ylabel("Late Delivery Rate (%)")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "late_rate_by_market.png"
)
plt.close()


# ------------------------------------------------------------
# 7. MONTHLY ORDER TREND
# ------------------------------------------------------------

print("\n6. MONTHLY ORDER VOLUME")

orders["order_month"] = (
    orders["order_date"].dt.to_period("M")
)

monthly_orders = (
    orders.groupby("order_month")
    .size()
    .reset_index(name="total_orders")
)

monthly_orders["order_month"] = (
    monthly_orders["order_month"].astype(str)
)

print(monthly_orders)

monthly_orders.to_csv(
    REPORT_PATH / "eda_monthly_orders.csv",
    index=False
)

plt.figure(figsize=(12, 5))

plt.plot(
    monthly_orders["order_month"],
    monthly_orders["total_orders"],
    marker="o"
)

plt.title("Monthly Order Volume")
plt.xlabel("Month")
plt.ylabel("Number of Orders")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "monthly_order_volume.png"
)
plt.close()


# ------------------------------------------------------------
# 8. ORDER CHARACTERISTICS
# ------------------------------------------------------------

print("\n7. ORDER CHARACTERISTICS")

order_summary = model_data[
    [
        "item_count",
        "total_quantity",
        "total_sales",
        "total_discount",
        "total_order_item_value",
        "average_discount_rate",
        "unique_products",
        "unique_categories"
    ]
].describe().round(2)

print(order_summary)

order_summary.to_csv(
    REPORT_PATH / "eda_order_characteristics.csv"
)


# ------------------------------------------------------------
# 9. CORRELATION ANALYSIS
# ------------------------------------------------------------

print("\n8. NUMERIC CORRELATION ANALYSIS")

correlation_columns = [
    "scheduled_days",
    "item_count",
    "total_quantity",
    "total_sales",
    "total_discount",
    "total_order_item_value",
    "average_discount_rate",
    "unique_products",
    "unique_categories",
    "late_delivery_risk"
]

correlation_matrix = model_data[
    correlation_columns
].corr()

print(correlation_matrix.round(2))

correlation_matrix.to_csv(
    REPORT_PATH / "eda_correlation_matrix.csv"
)

plt.figure(figsize=(11, 8))

plt.imshow(
    correlation_matrix,
    interpolation="nearest",
    aspect="auto"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(correlation_columns)),
    correlation_columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_columns)),
    correlation_columns
)

plt.title("Correlation Between Numeric Variables")
plt.tight_layout()

plt.savefig(
    FIGURE_PATH / "correlation_matrix.png"
)
plt.close()


# ------------------------------------------------------------
# 10. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 3 EDA COMPLETED")
print("=" * 60)

print("\nReports saved in:", REPORT_PATH)
print("Charts saved in:", FIGURE_PATH)

print("\nGenerated charts:")
for file in FIGURE_PATH.glob("*.png"):
    print("-", file.name)

print("\nEDA completed successfully.")
