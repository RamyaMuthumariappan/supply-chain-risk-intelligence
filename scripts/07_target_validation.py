
# ============================================================
# PHASE 7: TARGET LABEL VALIDATION
# Project: Supply Chain Risk Intelligence
# ============================================================

import os
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "orders_cleaned.csv"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

# Exclude cancelled orders
df = df[df["is_cancelled"] == 0].copy()

print("=" * 60)
print("PHASE 7: TARGET LABEL VALIDATION")
print("=" * 60)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = [
    "delay_days",
    "late_delivery_risk",
    "delivery_status",
    "shipping_mode",
    "scheduled_days"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(f"Missing columns: {missing}")

# ------------------------------------------------------------
# 4. CREATE AN INDEPENDENT LATE DELIVERY INDICATOR
# ------------------------------------------------------------

# Positive delay means actual delivery took longer
# than the scheduled delivery duration.

df["observed_late"] = (
    df["delay_days"] > 0
).astype(int)

print("\nOriginal target distribution:")
print(df["late_delivery_risk"].value_counts())

print("\nObserved late delivery distribution:")
print(df["observed_late"].value_counts())

# ------------------------------------------------------------
# 5. COMPARE ORIGINAL TARGET WITH OBSERVED DELAY
# ------------------------------------------------------------

comparison = pd.crosstab(
    df["late_delivery_risk"],
    df["observed_late"],
    rownames=["Original Risk Label"],
    colnames=["Observed Late (delay_days > 0)"]
)

print("\nOriginal target vs observed delivery delay:")
print(comparison)

comparison.to_csv(
    os.path.join(
        REPORT_DIR,
        "phase7_target_comparison.csv"
    )
)

# ------------------------------------------------------------
# 6. IDENTIFY LABEL DISAGREEMENTS
# ------------------------------------------------------------

mismatches = df[
    df["late_delivery_risk"] != df["observed_late"]
].copy()

print("\nNumber of mismatched orders:", len(mismatches))

print(
    "Mismatch percentage:",
    round(len(mismatches) / len(df) * 100, 2),
    "%"
)

print("\nSample mismatched orders:")

sample_columns = [
    "delay_days",
    "late_delivery_risk",
    "observed_late",
    "delivery_status",
    "shipping_mode",
    "scheduled_days"
]

print(
    mismatches[sample_columns]
    .head(15)
    .to_string(index=False)
)

mismatches[sample_columns].to_csv(
    os.path.join(
        REPORT_DIR,
        "phase7_mismatched_orders.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 7. CHECK ORIGINAL LABEL AGAINST DELIVERY STATUS
# ------------------------------------------------------------

status_comparison = pd.crosstab(
    df["delivery_status"],
    df["late_delivery_risk"]
)

print("\nDelivery Status vs Original Risk Label:")
print(status_comparison)

status_comparison.to_csv(
    os.path.join(
        REPORT_DIR,
        "phase7_status_vs_risk.csv"
    )
)

# ------------------------------------------------------------
# 8. CHECK SHIPPING MODE AND SCHEDULED DAYS
# ------------------------------------------------------------

shipping_schedule = pd.crosstab(
    df["shipping_mode"],
    df["scheduled_days"]
)

print("\nShipping Mode vs Scheduled Days:")
print(shipping_schedule)

shipping_schedule.to_csv(
    os.path.join(
        REPORT_DIR,
        "phase7_shipping_schedule_relationship.csv"
    )
)

# ------------------------------------------------------------
# 9. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 7 TARGET VALIDATION COMPLETED")
print("=" * 60)

print("\nReports generated:")
print("- phase7_target_comparison.csv")
print("- phase7_mismatched_orders.csv")
print("- phase7_status_vs_risk.csv")
print("- phase7_shipping_schedule_relationship.csv")
