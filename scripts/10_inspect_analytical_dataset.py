
import pandas as pd
from pathlib import Path

# Load the Phase 9 analytical dataset
file_path = Path("data/processed/orders_analytics.csv")

df = pd.read_csv(file_path, encoding="latin1")

print("=" * 60)
print("ANALYTICAL DATASET - COLUMN INSPECTION")
print("=" * 60)

print("\nDataset shape:", df.shape)

print("\nColumn names and data types:")
for column in df.columns:
    print(f"{column:<35} {df[column].dtype}")

print("\nFirst 5 rows:")
print(df.head())

print("\nDuplicate order IDs:", df["order_id"].duplicated().sum())
print("=" * 60)
