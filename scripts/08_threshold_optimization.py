import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    balanced_accuracy_score,
    confusion_matrix
)


# --------------------------------------------------
# 1. PATHS AND DATA
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders_cleaned.csv"
)

REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Convert order_date into datetime
df["order_date"] = pd.to_datetime(df["order_date"])

# Exclude cancelled orders
df = df[df["is_cancelled"] == 0].copy()

# CREATE DATE FEATURES BEFORE FEATURE SELECTION
df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_day_of_week"] = df["order_date"].dt.dayofweek

# Verify that the columns exist
print("\nDate features created:")
print(df[[
    "order_date",
    "order_year",
    "order_month",
    "order_day_of_week"
]].head())



# --------------------------------------------------
# 2. FEATURE SELECTION
# --------------------------------------------------

# Keep shipping_mode and remove scheduled_days
# because they are perfectly linked in this dataset.

categorical_features = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment"
]

numerical_features = [
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

features = categorical_features + numerical_features
target = "late_delivery_risk"

X = df[features].copy()
y = df[target].copy()

# --------------------------------------------------
# 3. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

test_start = pd.Timestamp("2017-06-20")

pre_test_dates = df.loc[
    df["order_date"] < test_start, "order_date"
].sort_values()

# Use the 80th percentile date of the pre-test period
validation_start = pre_test_dates.quantile(0.80).normalize()

train_mask = df["order_date"] < validation_start

validation_mask = (
    (df["order_date"] >= validation_start) &
    (df["order_date"] < test_start)
)

test_mask = df["order_date"] >= test_start

X_train = X.loc[train_mask]
y_train = y.loc[train_mask]

X_val = X.loc[validation_mask]
y_val = y.loc[validation_mask]

X_test = X.loc[test_mask]
y_test = y.loc[test_mask]

print("=" * 60)
print("PHASE 8: THRESHOLD OPTIMISATION")
print("=" * 60)

print("\nValidation begins:", validation_start.date())
print("Test begins:", test_start.date())

print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("categorical", categorical_transformer, categorical_features),
    ("numerical", numerical_transformer, numerical_features)
])

# --------------------------------------------------
# 5. MODELS
# --------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
}

results = []

# --------------------------------------------------
# 6. TRAIN, SELECT THRESHOLD, TEST
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train only on the earliest period
    pipeline.fit(X_train, y_train)

    # Select threshold using validation data
    validation_probabilities = pipeline.predict_proba(X_val)[:, 1]

    threshold_results = []

    for threshold in np.arange(0.05, 0.96, 0.01):

        validation_predictions = (
            validation_probabilities >= threshold
        ).astype(int)

        threshold_results.append({
    "threshold": threshold,
    "precision": precision_score(
        y_val, validation_predictions, zero_division=0
    ),
    "recall": recall_score(
        y_val, validation_predictions, zero_division=0
    ),
    "f1": f1_score(
        y_val, validation_predictions, zero_division=0
    ),
    "balanced_accuracy": balanced_accuracy_score(
        y_val, validation_predictions
    )
})

    threshold_df = pd.DataFrame(threshold_results)

    # Select the threshold with the highest validation F1
    best_row = threshold_df.loc[
    threshold_df["balanced_accuracy"].idxmax()
]
    print(
    "Validation balanced accuracy:",
    round(best_row["balanced_accuracy"], 4)
)
    best_threshold = best_row["threshold"]

    print("\nBest validation threshold:", round(best_threshold, 2))
    print("Validation precision:", round(best_row["precision"], 4))
    print("Validation recall:", round(best_row["recall"], 4))
    print("Validation F1:", round(best_row["f1"], 4))

    # Evaluate once on the untouched test period
    test_probabilities = pipeline.predict_proba(X_test)[:, 1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    cm = confusion_matrix(y_test, test_predictions)

    accuracy = accuracy_score(y_test, test_predictions)
    precision = precision_score(
        y_test, test_predictions, zero_division=0
    )
    recall = recall_score(
        y_test, test_predictions, zero_division=0
    )
    f1 = f1_score(y_test, test_predictions, zero_division=0)
    auc = roc_auc_score(y_test, test_probabilities)
    balanced_acc = balanced_accuracy_score(y_test, test_predictions)

    print("\nTest results:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1:", round(f1, 4))
    print("ROC-AUC:", round(auc, 4))
    print("Balanced Accuracy:", round(balanced_acc, 4))

    print("\nConfusion Matrix:")
    print(cm)

    results.append({
        "model": model_name,
        "best_threshold": best_threshold,
        "validation_f1": best_row["f1"],
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
        "test_roc_auc": auc,
        "test_balanced_accuracy": balanced_acc
    })

# --------------------------------------------------
# 7. SAVE RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

output_path = os.path.join(
    REPORT_DIR, "phase8_threshold_optimization.csv"
)

results_df.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("PHASE 8 COMPLETED")
print("=" * 60)

print("\nModel comparison:")
print(results_df.to_string(index=False))

print("\nReport saved:", output_path)