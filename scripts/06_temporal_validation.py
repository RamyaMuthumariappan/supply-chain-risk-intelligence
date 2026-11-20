
# ============================================================
# PHASE 6: FEATURE LEAKAGE INVESTIGATION
#          AND TIME-BASED MODEL VALIDATION
#
# Project: Supply Chain Risk Intelligence
# ============================================================

import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


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

# Exclude cancelled orders, as in Phase 4
df = df[df["is_cancelled"] == 0].copy()

print("Dataset shape after excluding cancelled orders:", df.shape)

print("=" * 65)
print("PHASE 6: FEATURE LEAKAGE AND TEMPORAL VALIDATION")
print("=" * 65)

print("\nDataset shape:", df.shape)


# ------------------------------------------------------------
# 3. CHECK ORDER DATE
# ------------------------------------------------------------

df["order_date"] = pd.to_datetime(
    df["order_date"],
    errors="coerce"
)

invalid_dates = df["order_date"].isna().sum()

print("\nInvalid order dates:", invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Some order dates could not be parsed. "
        "Investigate these before temporal validation."
    )
    

# Recreate date-based features used in Phase 4

df["order_year"] = df["order_date"].dt.year

df["order_month"] = df["order_date"].dt.month

df["order_day_of_week"] = df["order_date"].dt.dayofweek

print("\nDate-based features recreated:")
print(
    df[
        ["order_date", "order_year", "order_month", "order_day_of_week"]
    ].head()
)


print("Earliest order date:", df["order_date"].min())
print("Latest order date:", df["order_date"].max())


# ------------------------------------------------------------
# 4. DEFINE FEATURES AND TARGET
# ------------------------------------------------------------

categorical_features = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment"
]

numerical_features = [
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

target_column = "late_delivery_risk"

all_features = categorical_features + numerical_features

required_columns = all_features + [
    target_column,
    "order_date"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

X = df[all_features].copy()
y = df[target_column].copy()

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(
    (y.value_counts(normalize=True) * 100).round(2)
)


# ------------------------------------------------------------
# 5. FEATURE LEAKAGE AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE LEAKAGE AUDIT")
print("=" * 65)

# These are outcome-related fields that must not be
# included as predictors for an order-time prediction model.

post_event_columns = [
    "delivery_status",
    "order_status",
    "days_for_shipping_real",
    "shipping_date",
    "actual_delivery_date",
    "delay_days"
]

print("\nChecking for post-event fields in predictors:")

for col in post_event_columns:

    if col in X.columns:
        print(f"WARNING: {col} is included in predictors!")

    else:
        print(f"PASS: {col} is not included in predictors.")

# Check whether the original source label has been
# accidentally included in the predictor columns.

if target_column in X.columns:
    raise ValueError("Target leakage: target is inside X!")

print("\nTarget column is excluded from predictors.")


# ------------------------------------------------------------
# 6. INVESTIGATE IMPORTANT FEATURE RELATIONSHIPS
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("FEATURE AND TARGET RELATIONSHIPS")
print("=" * 65)

# Late delivery rate by shipping mode

shipping_analysis = (
    df.groupby("shipping_mode")[target_column]
    .agg(
        total_orders="count",
        late_orders="sum",
        late_rate="mean"
    )
    .reset_index()
)

shipping_analysis["late_rate"] *= 100

print("\nLate delivery rate by shipping mode:")
print(shipping_analysis.to_string(index=False))

shipping_analysis.to_csv(
    os.path.join(
        REPORT_DIR,
        "phase6_late_rate_by_shipping_mode.csv"
    ),
    index=False
)


# Late delivery rate by scheduled days

scheduled_analysis = (
    df.groupby("scheduled_days")[target_column]
    .agg(
        total_orders="count",
        late_orders="sum",
        late_rate="mean"
    )
    .reset_index()
)

scheduled_analysis["late_rate"] *= 100

print("\nLate delivery rate by scheduled days:")
print(scheduled_analysis.to_string(index=False))

scheduled_analysis.to_csv(
    os.path.join(
        REPORT_DIR,
        "phase6_late_rate_by_scheduled_days.csv"
    ),
    index=False
)


# Check redundancy between unique_products and unique_categories

correlation = df[
    ["unique_products", "unique_categories"]
].corr().iloc[0, 1]

print(
    "\nCorrelation between unique_products "
    "and unique_categories:",
    round(correlation, 4)
)



# ------------------------------------------------------------
# 7. CHRONOLOGICAL TRAIN/TEST SPLIT
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("CHRONOLOGICAL TRAIN/TEST SPLIT")
print("=" * 65)

# Group orders by calendar date, not individual timestamps

df["order_calendar_date"] = df["order_date"].dt.normalize()

unique_dates = np.sort(
    df["order_calendar_date"].unique()
)

split_position = int(len(unique_dates) * 0.80)

split_position = min(
    max(split_position, 1),
    len(unique_dates) - 1
)

cutoff_date = pd.Timestamp(
    unique_dates[split_position]
)

train_mask = df["order_calendar_date"] < cutoff_date
test_mask = df["order_calendar_date"] >= cutoff_date

X_train = X.loc[train_mask].copy()
X_test = X.loc[test_mask].copy()

y_train = y.loc[train_mask].copy()
y_test = y.loc[test_mask].copy()

print("\nCutoff date:", cutoff_date)

print("\nTraining period:")
print(df.loc[train_mask, "order_date"].min())
print("to")
print(df.loc[train_mask, "order_date"].max())

print("\nTesting period:")
print(df.loc[test_mask, "order_date"].min())
print("to")
print(df.loc[test_mask, "order_date"].max())

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).round(3))

print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).round(3))

if y_train.nunique() < 2 or y_test.nunique() < 2:
    raise ValueError(
        "Training or testing data contains only one target class."
    )



# ------------------------------------------------------------
# 8. DEFINE FEATURE GROUP EXPERIMENTS
# ------------------------------------------------------------

feature_experiments = {

    "A_All_Features": all_features,

    "B_Without_Shipping_Mode": [
        col for col in all_features
        if col != "shipping_mode"
    ],

    "C_Without_Scheduled_Days": [
        col for col in all_features
        if col != "scheduled_days"
    ],

    "D_Without_Both": [
        col for col in all_features
        if col not in [
            "shipping_mode",
            "scheduled_days"
        ]
    ],

    "E_Without_Unique_Categories": [
        col for col in all_features
        if col != "unique_categories"
    ]
}


# ------------------------------------------------------------
# 9. CREATE MODEL PIPELINE
# ------------------------------------------------------------

def create_pipeline(model, features):

    categorical = [
        col for col in features
        if col in categorical_features
    ]

    numerical = [
        col for col in features
        if col in numerical_features
    ]

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    numerical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                categorical
            ),
            (
                "numerical",
                numerical_transformer,
                numerical
            )
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return pipeline


# ------------------------------------------------------------
# 10. BASELINE MODEL
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("BASELINE MODEL")
print("=" * 65)

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(X_train, y_train)

baseline_predictions = baseline.predict(X_test)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)

print(
    "\nMajority-class baseline accuracy:",
    round(baseline_accuracy, 4)
)

print(
    "Baseline confusion matrix:\n",
    confusion_matrix(y_test, baseline_predictions)
)


# ------------------------------------------------------------
# 11. TRAIN AND EVALUATE FEATURE EXPERIMENTS
# ------------------------------------------------------------

results = []

model_definitions = {

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


for experiment_name, features in feature_experiments.items():

    print("\n" + "=" * 65)
    print("EXPERIMENT:", experiment_name)
    print("=" * 65)

    print("Number of input features:", len(features))

    for model_name, model_definition in model_definitions.items():

        print("\nModel:", model_name)

        pipeline = create_pipeline(
            model_definition,
            features
        )

        pipeline.fit(
            X_train[features],
            y_train
        )

        predictions = pipeline.predict(
            X_test[features]
        )

        probabilities = pipeline.predict_proba(
            X_test[features]
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test,
            probabilities
        )

        print("Accuracy:", round(accuracy, 4))
        print("Precision:", round(precision, 4))
        print("Recall:", round(recall, 4))
        print("F1-score:", round(f1, 4))
        print("ROC-AUC:", round(auc, 4))

        print("\nConfusion matrix:")
        print(confusion_matrix(y_test, predictions))

        results.append({

            "Experiment": experiment_name,
            "Model": model_name,
            "Number_of_Features": len(features),
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1_Score": f1,
            "ROC_AUC": auc

        })


# ------------------------------------------------------------
# 12. SAVE FINAL EXPERIMENT COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC_AUC",
    ascending=False
)

results_df.to_csv(
    os.path.join(
        REPORT_DIR,
        "phase6_temporal_validation_results.csv"
    ),
    index=False
)

print("\n" + "=" * 65)
print("FINAL TEMPORAL VALIDATION COMPARISON")
print("=" * 65)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ------------------------------------------------------------
# 13. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE 6 COMPLETED")
print("=" * 65)

print("\nBaseline accuracy:", round(baseline_accuracy, 4))

print("\nBest experiment by ROC-AUC:")

print(
    results_df.iloc[0][
        [
            "Experiment",
            "Model",
            "ROC_AUC",
            "Recall",
            "Precision",
            "F1_Score"
        ]
    ]
)

print("\nReport saved:")
print("reports/phase6_temporal_validation_results.csv")

print("\nAdditional reports saved:")
print("- phase6_late_rate_by_shipping_mode.csv")
print("- phase6_late_rate_by_scheduled_days.csv")
