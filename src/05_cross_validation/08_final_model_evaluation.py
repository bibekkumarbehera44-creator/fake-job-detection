
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

FEATURE_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "features"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RESULT_DIR = os.path.join(
    PROJECT_ROOT, "results", "final_evaluation"
)

os.makedirs(RESULT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD FINAL MODEL
# --------------------------------------------------

print("\nLoading final Linear SVM ROS model...")

model_path = os.path.join(
    MODEL_DIR, "linear_svm_ros.joblib"
)

final_model = joblib.load(model_path)

print("Model loaded successfully.")

# --------------------------------------------------
# 3. LOAD TEST DATA
# --------------------------------------------------

X_test = load_npz(
    os.path.join(FEATURE_DIR, "combined_test.npz")
)

y_test = pd.read_csv(
    os.path.join(FEATURE_DIR, "y_test.csv")
).iloc[:, 0].values

print("\nTest data shape:", X_test.shape)
print("Test target shape:", y_test.shape)

# --------------------------------------------------
# 4. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating predictions...")

y_scores = final_model.decision_function(X_test)

# Threshold selected using training-only cross-validation
threshold = -0.10383682637001107

y_predictions = (
    y_scores >= threshold
).astype(int)

print("Predictions generated successfully.")

# --------------------------------------------------
# 5. CALCULATE EVALUATION METRICS
# --------------------------------------------------

metrics = {
    "Accuracy": accuracy_score(y_test, y_predictions),
    "Precision": precision_score(
        y_test, y_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_predictions, zero_division=0
    ),
    "F1 Score": f1_score(
        y_test, y_predictions, zero_division=0
    ),
    "ROC-AUC": roc_auc_score(y_test, y_scores),
    "PR-AUC": average_precision_score(y_test, y_scores)
}

print("\nFINAL MODEL PERFORMANCE")
print("=" * 50)

for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# --------------------------------------------------
# 6. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, y_predictions)

print("\nConfusion Matrix:")
print(cm)

# --------------------------------------------------
# 7. CLASSIFICATION REPORT
# --------------------------------------------------

report = classification_report(
    y_test,
    y_predictions,
    target_names=["Genuine", "Fraudulent"],
    zero_division=0
)

print("\nClassification Report:")
print(report)

# --------------------------------------------------
# 8. SAVE FINAL RESULTS
# --------------------------------------------------

metrics_df = pd.DataFrame([metrics])

metrics_df.to_csv(
    os.path.join(RESULT_DIR, "final_model_metrics.csv"),
    index=False
)

with open(
    os.path.join(RESULT_DIR, "classification_report.txt"),
    "w",
    encoding="utf-8"
) as file:
    file.write(report)

np.savetxt(
    os.path.join(RESULT_DIR, "confusion_matrix.csv"),
    cm,
    delimiter=",",
    fmt="%d"
)

print("\nAll final evaluation results saved successfully.")

print("\nResults folder:")
print(RESULT_DIR)

print("\nFINAL MODEL EVALUATION COMPLETED.")

# --------------------------------------------------
# 9. GENERATE CONFUSION MATRIX GRAPH
# --------------------------------------------------

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - Linear SVM ROS")
plt.colorbar()

class_names = ["Genuine", "Fraudulent"]

plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)

for i in range(2):
    for j in range(2):
        plt.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 10. GENERATE MODEL PERFORMANCE GRAPH
# --------------------------------------------------

plt.figure(figsize=(10, 6))

metric_names = list(metrics.keys())
metric_values = list(metrics.values())

bars = plt.bar(metric_names, metric_values)

plt.title("Final Model Performance - Linear SVM ROS")
plt.xlabel("Evaluation Metrics")
plt.ylabel("Score")
plt.ylim(0, 1.05)

plt.xticks(rotation=25)

for bar, value in zip(bars, metric_values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.01,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(RESULT_DIR, "final_model_performance.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nConfusion matrix graph saved.")
print("Final model performance graph saved.")

