
# ============================================================
# PHASE 5: MODEL EVALUATION AND INTERPRETATION
# Project: Supply Chain Risk Intelligence
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

# ------------------------------------------------------------
# 1. SET PROJECT PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR, "data", "processed", "orders_for_prediction.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
FIGURE_DIR = os.path.join(REPORT_DIR, "figures")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. LOAD DATA AND SAVED MODELS
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

logistic_model = joblib.load(
    os.path.join(MODEL_DIR, "logistic_regression.pkl")
)

random_forest_model = joblib.load(
    os.path.join(MODEL_DIR, "random_forest.pkl")
)

print("=" * 60)
print("PHASE 5: MODEL EVALUATION")
print("=" * 60)

print("\nDataset shape:", df.shape)

# ------------------------------------------------------------
# 3. DEFINE FEATURES AND TARGET
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

feature_columns = categorical_features + numerical_features
target_column = "late_delivery_risk"

X = df[feature_columns]
y = df[target_column]

# Reproduce the same split used in Phase 4
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# 4. EVALUATE BOTH MODELS
# ------------------------------------------------------------

models = {
    "Logistic Regression": logistic_model,
    "Random Forest": random_forest_model
}

evaluation_results = []

predictions = {}

for model_name, model in models.items():

    print("\n" + "=" * 60)
    print(model_name.upper())
    print("=" * 60)

    # Predictions
    y_pred = model.predict(X_test)

    # Predicted probabilities for class 1
    y_probability = model.predict_proba(X_test)[:, 1]

    predictions[model_name] = {
        "predicted": y_pred,
        "probability": y_probability
    }

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_probability)

    print(f"\nAccuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    pd.DataFrame(report).transpose().to_csv(
        os.path.join(
            REPORT_DIR,
            model_name.lower().replace(" ", "_") + "_classification_report.csv"
        )
    )

    evaluation_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "ROC_AUC": auc
    })

    # --------------------------------------------------------
    # 5. CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    cm_df = pd.DataFrame(
        cm,
        index=["Actual Not Late", "Actual Late"],
        columns=["Predicted Not Late", "Predicted Late"]
    )

    cm_df.to_csv(
        os.path.join(
            REPORT_DIR,
            model_name.lower().replace(" ", "_") + "_confusion_matrix.csv"
        )
    )

    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.colorbar()

    plt.xticks([0, 1], ["Not Late", "Late"])
    plt.yticks([0, 1], ["Not Late", "Late"])

    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        ),
        dpi=300
    )

    plt.close()


# ------------------------------------------------------------
# 6. SAVE MODEL COMPARISON
# ------------------------------------------------------------

comparison_df = pd.DataFrame(evaluation_results)

comparison_df.to_csv(
    os.path.join(REPORT_DIR, "phase5_model_evaluation.csv"),
    index=False
)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison_df.to_string(index=False))


# ------------------------------------------------------------
# 7. ROC CURVE COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

for model_name, result in predictions.items():

    fpr, tpr, thresholds = roc_curve(
        y_test,
        result["probability"]
    )

    auc = roc_auc_score(y_test, result["probability"])

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC = {auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "roc_curve_comparison.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 8. RANDOM FOREST FEATURE IMPORTANCE
# ------------------------------------------------------------

# The pipeline's preprocessing step transforms the features.
# Get the names of those transformed features.
rf_feature_names = random_forest_model[:-1].get_feature_names_out()

rf_classifier = random_forest_model[-1]

rf_importances = rf_classifier.feature_importances_

rf_importance_df = pd.DataFrame({
    "Feature": rf_feature_names,
    "Importance": rf_importances
})

rf_importance_df = rf_importance_df.sort_values(
    by="Importance",
    ascending=False
)

rf_importance_df.to_csv(
    os.path.join(REPORT_DIR, "random_forest_feature_importance.csv"),
    index=False
)

print("\nTop 20 Random Forest Features:")
print(rf_importance_df.head(20).to_string(index=False))

# Plot top 20 features
top_features = rf_importance_df.head(20).sort_values(
    by="Importance"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Feature Importance")
plt.title("Top 20 Random Forest Features")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "random_forest_feature_importance.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 9. LOGISTIC REGRESSION COEFFICIENTS
# ------------------------------------------------------------

lr_feature_names = logistic_model[:-1].get_feature_names_out()

lr_classifier = logistic_model[-1]

lr_coefficients = lr_classifier.coef_[0]

lr_coefficient_df = pd.DataFrame({
    "Feature": lr_feature_names,
    "Coefficient": lr_coefficients
})

lr_coefficient_df["Absolute_Coefficient"] = (
    lr_coefficient_df["Coefficient"].abs()
)

lr_coefficient_df = lr_coefficient_df.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

lr_coefficient_df.to_csv(
    os.path.join(REPORT_DIR, "logistic_regression_coefficients.csv"),
    index=False
)

print("\nTop 20 Logistic Regression Coefficients:")
print(lr_coefficient_df.head(20).to_string(index=False))

# Plot top 20 coefficients, showing their direction
top_coefficients = lr_coefficient_df.head(20).sort_values(
    by="Coefficient"
)

plt.figure(figsize=(10, 8))

plt.barh(
    top_coefficients["Feature"],
    top_coefficients["Coefficient"]
)

plt.axvline(x=0, linestyle="--")

plt.xlabel("Coefficient")
plt.title("Top 20 Logistic Regression Coefficients")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "logistic_regression_coefficients.png"),
    dpi=300
)

plt.close()


# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("PHASE 5 COMPLETED")
print("=" * 60)

print("\nGenerated reports:")
print("- phase5_model_evaluation.csv")
print("- logistic_regression_classification_report.csv")
print("- random_forest_classification_report.csv")
print("- logistic_regression_confusion_matrix.csv")
print("- random_forest_confusion_matrix.csv")
print("- random_forest_feature_importance.csv")
print("- logistic_regression_coefficients.csv")

print("\nGenerated figures:")
print("- roc_curve_comparison.png")
print("- logistic_regression_confusion_matrix.png")
print("- random_forest_confusion_matrix.png")
print("- random_forest_feature_importance.png")
print("- logistic_regression_coefficients.png")

print("\nCheck the reports and figures folders.")
