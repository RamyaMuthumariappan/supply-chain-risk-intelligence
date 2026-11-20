
import os
import pandas as pd

# --------------------------------------------------
# 1. PATHS
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders_cleaned.csv"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders_analytics.csv"
)

# --------------------------------------------------
# 2. LOAD DATA
# --------------------------------------------------

df = pd.read_csv(INPUT_PATH)

# Standardise column names
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
)

print("\nAvailable columns:")
print(df.columns.tolist())

print("=" * 60)
print("PHASE 9: ANALYTICAL DATASET PREPARATION")
print("=" * 60)

print("\nOriginal dataset shape:", df.shape)

# --------------------------------------------------
# 3. REMOVE CUSTOMER IDENTIFIERS
# --------------------------------------------------

sensitive_columns = [
    "customer_id",
    "order_customer_id",
    "customer_fname",
    "customer_lname",
    "customer_email",
    "customer_password",
    "customer_street",
    "customer_zipcode",
    "order_zipcode"
]

df.drop(
    columns=sensitive_columns,
    errors="ignore",
    inplace=True
)

# --------------------------------------------------
# 4. DATE CONVERSION
# --------------------------------------------------

df["order_date"] = pd.to_datetime(
    df["order_date"], errors="coerce"
)

# Convert optional date columns if present
for column in ["shipping_date", "actual_delivery_date"]:
    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column], errors="coerce"
        )

# --------------------------------------------------
# 5. DATE FEATURE ENGINEERING
# --------------------------------------------------

df["order_year"] = df["order_date"].dt.year

df["order_month"] = df["order_date"].dt.month

df["order_month_name"] = df["order_date"].dt.month_name()

df["order_quarter"] = df["order_date"].dt.quarter

df["order_day_of_week"] = df["order_date"].dt.dayofweek

df["order_day_name"] = df["order_date"].dt.day_name()

# --------------------------------------------------
# 6. DELIVERY OUTCOME VALIDATION
# --------------------------------------------------

df["delivery_outcome"] = df["delivery_status"]

# Check the essential columns
required_columns = [
    "order_id",
    "order_date",
    "delivery_status",
    "late_delivery_risk",
    "is_cancelled",
    "delay_days"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )

# --------------------------------------------------
# 7. DATA QUALITY CHECKS
# --------------------------------------------------

print("\nDataset shape after preparation:", df.shape)

print("\nDuplicate order IDs:", df["order_id"].duplicated().sum())

print("\nMissing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nCancelled orders:")
print(df["is_cancelled"].value_counts())

print("\nDelivery outcome:")
print(df["delivery_outcome"].value_counts())

print("\nLate delivery risk:")
print(df["late_delivery_risk"].value_counts())

# Validate order-level grain
if df["order_id"].duplicated().any():
    raise ValueError("Duplicate Order IDs detected!")

if df["order_date"].isnull().any():
    raise ValueError("Invalid or missing order dates detected!")

# --------------------------------------------------
# 8. EXPORT ANALYTICAL DATASET
# --------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 60)
print("PHASE 9 COMPLETED")
print("=" * 60)

print("\nFinal dataset shape:", df.shape)
print("\nSaved to:", OUTPUT_PATH)
