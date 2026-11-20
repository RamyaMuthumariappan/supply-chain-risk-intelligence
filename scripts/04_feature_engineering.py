# ============================================================
# SUPPLY CHAIN RISK INTELLIGENCE
# Phase 4: Feature Engineering and ML Preparation
# ============================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

import joblib

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "data" / "processed" /
    "orders_for_prediction.csv"
)

REPORT_PATH = PROJECT_DIR / "reports"
MODEL_PATH = PROJECT_DIR / "models"

REPORT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("PHASE 4: FEATURE ENGINEERING AND ML PREPARATION")
print("=" * 65)

print("\nDataset shape:", df.shape)

print("\nDataset columns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum().sum())


# ------------------------------------------------------------
# 3. DEFINE TARGET AND FEATURES
# ------------------------------------------------------------

TARGET = "late_delivery_risk"

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

features = categorical_features + numerical_features

# Check that all expected columns exist
missing_columns = [
    column for column in features + [TARGET]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing expected columns: {missing_columns}"
    )

X = df[features].copy()
y = df[TARGET].copy()

print("\nFeature count:", len(features))
print("Categorical features:", len(categorical_features))
print("Numerical features:", len(numerical_features))

print("\nTarget distribution:")
print(y.value_counts())

print("\nTarget percentages:")
print(y.value_counts(normalize=True).mul(100).round(2))


# ------------------------------------------------------------
# 4. CHECK FEATURE CORRELATIONS
# ------------------------------------------------------------

print("\n1. FEATURE REDUNDANCY CHECK")

print(
    "\nUnique products vs unique categories:"
)

print(
    X[["unique_products", "unique_categories"]]
    .corr()
)

print(
    "\nThese features will initially be retained."
    "\nWe can compare models with and without redundant "
    "features later."
)


# ------------------------------------------------------------
# 5. TRAIN-TEST SPLIT
# ------------------------------------------------------------

print("\n2. TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training records:", X_train.shape[0])
print("Testing records:", X_test.shape[0])

print(
    "\nTraining target distribution:"
)

print(
    y_train.value_counts(normalize=True).mul(100).round(2)
)

print(
    "\nTesting target distribution:"
)

print(
    y_test.value_counts(normalize=True).mul(100).round(2)
)


# ------------------------------------------------------------
# 6. PREPROCESSING
# ------------------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ------------------------------------------------------------
# 7. DEFINE MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
}


# ------------------------------------------------------------
# 8. TRAIN AND EVALUATE MODELS
# ------------------------------------------------------------

results = []
trained_models = {}

for model_name, classifier in models.items():

    print("\n" + "=" * 65)
    print("TRAINING:", model_name)
    print("=" * 65)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    print("\nModel performance:")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("ROC-AUC:", round(roc_auc, 4))

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": roc_auc
    })

    trained_models[model_name] = pipeline


# ------------------------------------------------------------
# 9. MODEL COMPARISON
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "F1_Score",
    ascending=False
)

print("\n" + "=" * 65)
print("MODEL COMPARISON")
print("=" * 65)

print(results_df.round(4))

results_df.to_csv(
    REPORT_PATH / "model_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 10. SAVE TRAINED MODELS
# ------------------------------------------------------------

for model_name, pipeline in trained_models.items():

    filename = (
        model_name.lower().replace(" ", "_") + ".pkl"
    )

    joblib.dump(
        pipeline,
        MODEL_PATH / filename
    )

    print("\nSaved model:", filename)


# ------------------------------------------------------------
# 11. COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("PHASE 4 COMPLETED")
print("=" * 65)

print("\nModel comparison saved in:", REPORT_PATH)
print("Trained models saved in:", MODEL_PATH)

print("\nMachine learning preparation completed.")