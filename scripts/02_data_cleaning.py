
# ============================================================
# SUPPLY CHAIN RISK INTELLIGENCE
# Phase 2: Data Cleaning and Order-Level Transformation
# ============================================================

# 1. Import libraries

import pandas as pd
import numpy as np
from pathlib import Path

# 2. Define project paths

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = (
    PROJECT_DIR / "data" / "raw" /
    "DataCoSupplyChainDataset.csv"
)

PROCESSED_DATA_PATH = PROJECT_DIR / "data" / "processed"

REPORT_PATH = PROJECT_DIR / "reports"

PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

# 3. Load the original dataset

df = pd.read_csv(
    RAW_DATA_PATH,
    encoding="latin1",
    low_memory=False
)

print("=" * 60)
print("PHASE 2: DATA CLEANING")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

# 4. Remove irrelevant and sensitive columns

columns_to_drop = [
    "Product Description",  # 100% missing
    "Order Zipcode",        # 86.24% missing
    "Customer Lname",
    "Customer Fname",
    "Customer Email",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",
    "Customer Id",
    "Product Image",
    "Product Status"
]

df = df.drop(
    columns=columns_to_drop,
    errors="ignore"
)

print("\nColumns removed:", len(columns_to_drop))
print("Dataset shape after column removal:", df.shape)

# 5. Parse the order date

df["order date (DateOrders)"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

print(
    "\nInvalid order dates:",
    df["order date (DateOrders)"].isna().sum()
)

# 6. Check missing values after column removal

missing_summary = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().sum() / len(df)
    ) * 100
})

missing_summary = missing_summary[
    missing_summary["Missing_Count"] > 0
]

print("\nRemaining missing values:")
print(missing_summary)

missing_summary.to_csv(
    REPORT_PATH / "phase2_missing_values.csv"
)

# 7. Check order-level consistency

order_level_columns = [
    "Order Customer Id",
    "order date (DateOrders)",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Order State",
    "Order City",
    "Customer Segment",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Late_delivery_risk",
    "Order Status",
    "Days for shipping (real)"
]

consistency_report = df.groupby("Order Id")[
    order_level_columns
].nunique()

inconsistent_orders = (
    consistency_report > 1
).sum()

print("\nOrder-level consistency report:")
print(inconsistent_orders)

consistency_report.to_csv(
    REPORT_PATH / "order_consistency_report.csv"
)

# 8. Aggregate order-item records into orders

orders = df.groupby("Order Id").agg(

    # Order-level information
    customer_id=("Order Customer Id", "first"),
    order_date=("order date (DateOrders)", "first"),
    shipping_mode=("Shipping Mode", "first"),
    market=("Market", "first"),
    order_region=("Order Region", "first"),
    order_country=("Order Country", "first"),
    order_state=("Order State", "first"),
    order_city=("Order City", "first"),
    customer_segment=("Customer Segment", "first"),
    scheduled_days=("Days for shipment (scheduled)", "first"),

    # Delivery outcome information
    actual_shipping_days=("Days for shipping (real)", "first"),
    delivery_status=("Delivery Status", "first"),
    late_delivery_risk=("Late_delivery_risk", "first"),
    order_status=("Order Status", "first"),

    # Financial information
    benefit_per_order=("Benefit per order", "first"),
    sales_per_customer=("Sales per customer", "first"),
    order_profit=("Order Profit Per Order", "first"),

    # Item-level information aggregated to order level
    item_count=("Order Item Id", "count"),
    total_quantity=("Order Item Quantity", "sum"),
    total_sales=("Sales", "sum"),
    total_discount=("Order Item Discount", "sum"),
    total_order_item_value=("Order Item Total", "sum"),
    average_discount_rate=("Order Item Discount Rate", "mean"),
    unique_products=("Product Card Id", "nunique"),
    unique_categories=("Category Name", "nunique")

).reset_index()

print("\nOrder-level dataset created.")
print("Order-level dataset shape:", orders.shape)

# 9. Create delivery-related analytical variables

orders["delay_days"] = (
    orders["actual_shipping_days"]
    - orders["scheduled_days"]
)

orders["is_cancelled"] = (
    orders["delivery_status"] == "Shipping canceled"
).astype(int)

# 10. Check missing values in the aggregated dataset

print("\nMissing values in order-level dataset:")
print(
    orders.isnull().sum()[
        orders.isnull().sum() > 0
    ]
)

# 11. Check for duplicate orders

print(
    "\nDuplicate Order IDs:",
    orders["Order Id"].duplicated().sum()
)

print(
    "Unique orders after aggregation:",
    orders["Order Id"].nunique()
)

# 12. Inspect delivery outcomes at order level

print("\nDelivery status distribution:")
print(orders["delivery_status"].value_counts())

print("\nLate delivery risk distribution:")
print(orders["late_delivery_risk"].value_counts())

print("\nCancelled orders:")
print(orders["is_cancelled"].sum())

# 13. Save the complete cleaned order-level dataset

CLEANED_PATH = (
    PROCESSED_DATA_PATH / "orders_cleaned.csv"
)

orders.to_csv(
    CLEANED_PATH,
    index=False
)

print("\nCleaned order-level dataset saved.")

# ============================================================
# 14. Prepare a separate dataset for ML prediction
# ============================================================

# Exclude cancelled orders from the binary late-delivery model.
# They are not equivalent to successfully delivered orders.

model_data = orders[
    orders["is_cancelled"] == 0
].copy()

# Create date-based features

model_data["order_year"] = (
    model_data["order_date"].dt.year
)

model_data["order_month"] = (
    model_data["order_date"].dt.month
)

model_data["order_day_of_week"] = (
    model_data["order_date"].dt.dayofweek
)

# Select features available at order time.
# Do NOT include actual shipping days, delay days,
# delivery status or order status.

model_features = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment",
    "scheduled_days",
    "item_count",
    "total_quantity",
    "total_sales",
    "total_discount",
    "total_order_item_value",
    "average_discount_rate",
    "unique_products",
    "unique_categories",
    "order_year",
    "order_month",
    "order_day_of_week"
]

TARGET = "late_delivery_risk"

model_data = model_data[
    model_features + [TARGET]
].copy()

# Check missing values before saving

print("\nMissing values in ML dataset:")
print(
    model_data.isnull().sum()[
        model_data.isnull().sum() > 0
    ]
)

# Save the ML-ready dataset

MODEL_PATH = (
    PROCESSED_DATA_PATH / "orders_for_prediction.csv"
)

model_data.to_csv(
    MODEL_PATH,
    index=False
)

# 15. Final summary

print("\n" + "=" * 60)
print("PHASE 2 COMPLETED")
print("=" * 60)

print("\nCleaned order-level dataset:")
print(orders.shape)

print("\nML prediction dataset:")
print(model_data.shape)

print("\nTarget distribution:")
print(model_data[TARGET].value_counts())

print("\nTarget percentages:")
print(
    model_data[TARGET]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nSaved files:")
print(CLEANED_PATH)
print(MODEL_PATH)
print("\nData cleaning completed successfully.")
