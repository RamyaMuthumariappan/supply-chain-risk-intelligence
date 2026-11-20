
# ============================================================
# SUPPLY CHAIN RISK INTELLIGENCE
# Phase 1: Data Understanding
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

DICTIONARY_PATH = (
    PROJECT_DIR / "data" / "raw" /
    "DescriptionDataCoSupplyChain.csv"
)

REPORT_PATH = PROJECT_DIR / "reports"

REPORT_PATH.mkdir(parents=True, exist_ok=True)

# 3. Load datasets

df = pd.read_csv(
    RAW_DATA_PATH,
    encoding="latin1",
    low_memory=False
)

dictionary = pd.read_csv(
    DICTIONARY_PATH,
    encoding="latin1"
)

print("=" * 60)
print("DATASET LOADED SUCCESSFULLY")
print("=" * 60)

# 4. Basic dataset information

print("\nDataset shape:")
print(df.shape)

print("\nNumber of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 5. Inspect first and last records

print("\nFirst five records:")
print(df.head())

print("\nLast five records:")
print(df.tail())

# 6. Column names and data types

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

# 7. Dataset information

print("\nDataset information:")
df.info()

# 8. Statistical summary

print("\nNumerical summary:")
print(df.describe().T)

print("\nCategorical summary:")

string_columns = df.select_dtypes(
    include=["object"]
)

print(string_columns.describe().T)


# 9. Missing-value analysis

missing_values = pd.DataFrame({
    "Missing_Count": df.isnull().sum(),
    "Missing_Percentage": (
        df.isnull().sum() / len(df)
    ) * 100
})

missing_values = missing_values.sort_values(
    by="Missing_Percentage",
    ascending=False
)

print("\nMissing-value summary:")
print(missing_values.to_string())

# 10. Duplicate analysis

print("\nTotal duplicate rows:")
print(df.duplicated().sum())

print("\nUnique orders:")
print(df["Order Id"].nunique())

print("\nUnique order items:")
print(df["Order Item Id"].nunique())

# 11. Inspect delivery-related variables

delivery_columns = [
    "Shipping Mode",
    "Delivery Status",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk"
]

print("\nDelivery-related information:")

for column in delivery_columns:
    if column in df.columns:
        print(f"\n{column}")
        print(df[column].value_counts(dropna=False).head(15))

# 12. Inspect the data dictionary

print("\nData dictionary:")
print(dictionary.to_string(index=False))

# 13. Save missing-value report

missing_values.to_csv(
    REPORT_PATH / "missing_values_report.csv"
)

print("\nData understanding completed.")
print("Missing-value report saved successfully.")


# ============================================================
# ADDITIONAL INVESTIGATION: DELIVERY RISK
# ============================================================

# Calculate the difference between actual and scheduled days

df["Delay_Days"] = (
    df["Days for shipping (real)"]
    - df["Days for shipment (scheduled)"]
)

# Derive late status from shipping duration
df["Late_From_Duration"] = (
    df["Delay_Days"] > 0
).astype(int)

print("\nDelay days distribution:")
print(df["Delay_Days"].value_counts().sort_index())

print("\nDelivery risk versus actual delay:")
print(
    pd.crosstab(
        df["Late_delivery_risk"],
        df["Late_From_Duration"],
        margins=True
    )
)

print("\nDelivery status versus source risk:")
print(
    pd.crosstab(
        df["Delivery Status"],
        df["Late_delivery_risk"],
        margins=True
    )
)

print("\nOrder-level consistency check:")

order_consistency = df.groupby("Order Id").agg(
    Risk_Labels=("Late_delivery_risk", "nunique"),
    Delivery_Statuses=("Delivery Status", "nunique"),
    Actual_Days=("Days for shipping (real)", "nunique"),
    Scheduled_Days=("Days for shipment (scheduled)", "nunique")
)

print(order_consistency.describe())

print(
    "\nOrders with inconsistent risk labels:",
    (order_consistency["Risk_Labels"] > 1).sum()
)

print(
    "Orders with inconsistent delivery statuses:",
    (order_consistency["Delivery_Statuses"] > 1).sum()
)

