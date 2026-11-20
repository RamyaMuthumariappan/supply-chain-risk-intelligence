
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent

orders = pd.read_csv(
    PROJECT_DIR / "data" / "processed" / "orders_cleaned.csv"
)

# Exclude cancelled orders for delivery performance checks
delivered = orders[
    orders["delivery_status"] != "Shipping canceled"
].copy()

print("=" * 60)
print("EDA VALIDATION")
print("=" * 60)

# ------------------------------------------------------------
# 1. Compare source risk with calculated delivery delay
# ------------------------------------------------------------

delivered["calculated_late"] = (
    delivered["delay_days"] > 0
).astype(int)

print("\n1. SOURCE RISK VS CALCULATED DELAY")

risk_delay_table = pd.crosstab(
    delivered["late_delivery_risk"],
    delivered["calculated_late"],
    margins=True
)

print(risk_delay_table)

mismatches = (
    delivered["late_delivery_risk"]
    != delivered["calculated_late"]
).sum()

print("\nNumber of mismatched orders:", mismatches)

# ------------------------------------------------------------
# 2. Shipping mode and scheduled delivery days
# ------------------------------------------------------------

print("\n2. SHIPPING MODE VS SCHEDULED DAYS")

print(
    pd.crosstab(
        delivered["shipping_mode"],
        delivered["scheduled_days"]
    )
)

# ------------------------------------------------------------
# 3. Detailed shipping mode performance
# ------------------------------------------------------------

print("\n3. SHIPPING MODE PERFORMANCE")

shipping_detail = (
    delivered.groupby("shipping_mode")
    .agg(
        total_orders=("Order Id", "count"),
        late_orders=("late_delivery_risk", "sum"),
        late_rate=("late_delivery_risk", "mean"),
        average_actual_days=("actual_shipping_days", "mean"),
        average_scheduled_days=("scheduled_days", "mean"),
        average_delay=("delay_days", "mean"),
        minimum_delay=("delay_days", "min"),
        maximum_delay=("delay_days", "max")
    )
    .round(2)
)

shipping_detail["late_rate"] *= 100

print(shipping_detail)

# ------------------------------------------------------------
# 4. First Class delivery status and delay breakdown
# ------------------------------------------------------------

print("\n4. FIRST CLASS INVESTIGATION")

first_class = delivered[
    delivered["shipping_mode"] == "First Class"
]

print("\nFirst Class delivery status:")
print(first_class["delivery_status"].value_counts())

print("\nFirst Class delay distribution:")
print(first_class["delay_days"].value_counts().sort_index())

print("\nFirst Class risk distribution:")
print(first_class["late_delivery_risk"].value_counts())

print("\n" + "=" * 60)
print("VALIDATION COMPLETED")
print("=" * 60)
