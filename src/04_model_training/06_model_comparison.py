
import os
import warnings
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

FEATURE_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "features"
)

IMBALANCE_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed", "imbalance"
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RESULT_DIR = os.path.join(
    PROJECT_ROOT, "results", "model_comparison"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT, "results", "figures", "model_comparison"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD TRAINING AND TESTING DATA
# --------------------------------------------------

print("\nLoading dataset...")

X_train = load_npz(
    os.path.join(FEATURE_DIR, "combined_train.npz")
)

X_test = load_npz(
    os.path.join(FEATURE_DIR, "combined_test.npz")
)

y_train = pd.read_csv(
    os.path.join(FEATURE_DIR, "y_train.csv")
).iloc[:, 0].values

y_test = pd.read_csv(
    os.path.join(FEATURE_DIR, "y_test.csv")
).iloc[:, 0].values

print("Original training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining class distribution:")
print(pd.Series(y_train).value_counts().sort_index())

# Load oversampled training data
X_ros = load_npz(
    os.path.join(IMBALANCE_DIR, "combined_ros_train.npz")
)

y_ros = pd.read_csv(
    os.path.join(IMBALANCE_DIR, "y_ros_train.csv")
).iloc[:, 0].values

# Load SMOTE training data
X_smote = load_npz(
    os.path.join(IMBALANCE_DIR, "combined_smote_train.npz")
)

y_smote = pd.read_csv(
    os.path.join(IMBALANCE_DIR, "y_smote_train.csv")
).iloc[:, 0].values

# --------------------------------------------------
# 3. DEFINE MODELS
# --------------------------------------------------

experiments = {

    "Logistic Regression - Class Weight": (
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            solver="liblinear",
            random_state=42
        ),
        X_train,
        y_train
    ),

    "Logistic Regression - ROS": (
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=42
        ),
        X_ros,
        y_ros
    ),

    "Logistic Regression - SMOTE": (
        LogisticRegression(
            max_iter=2000,
            solver="liblinear",
            random_state=42
        ),
        X_smote,
        y_smote
    ),

    "Linear SVM - Class Weight": (
        LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ),
        X_train,
        y_train
    ),

    "Linear SVM - ROS": (
        LinearSVC(
            max_iter=5000,
            random_state=42
        ),
        X_ros,
        y_ros
    ),

    "Linear SVM - SMOTE": (
        LinearSVC(
            max_iter=5000,
            random_state=42
        ),
        X_smote,
        y_smote
    ),

    "Random Forest - Class Weight": (
        RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        X_train,
        y_train
    )
}

# Add XGBoost if it is installed
try:
    from xgboost import XGBClassifier

    negative_count = np.sum(y_train == 0)
    positive_count = np.sum(y_train == 1)

    experiments["XGBoost - Class Weight"] = (
        XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=negative_count / positive_count,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),
        X_train,
        y_train
    )

    print("\nXGBoost is available.")

except ImportError:
    print("\nXGBoost is not installed. Skipping XGBoost.")

# --------------------------------------------------
# 4. TRAIN AND EVALUATE MODELS
# --------------------------------------------------

results = []
trained_models = {}
prediction_scores = {}
model_predictions = {}

for model_name, (model, train_x, train_y) in experiments.items():

    print("\n" + "=" * 60)
    print("Training:", model_name)
    print("=" * 60)

    model.fit(train_x, train_y)

    # Predictions on the original untouched test data
    predictions = model.predict(X_test)

    # Obtain scores for ROC-AUC and PR-AUC
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC (Average Precision)": pr_auc
    })

    # Save trained model
    safe_name = re.sub(
        r"[^a-zA-Z0-9]+", "_", model_name
    ).strip("_").lower()

    model_path = os.path.join(
        MODEL_DIR, safe_name + ".joblib"
    )

    joblib.dump(model, model_path)

    trained_models[model_name] = model_path
    prediction_scores[model_name] = scores
    model_predictions[model_name] = predictions

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    print("ROC-AUC  :", round(roc_auc, 4))
    print("PR-AUC   :", round(pr_auc, 4))
    print("Model saved:", model_path)

# --------------------------------------------------
# 5. SAVE MODEL COMPARISON REPORT
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by=["F1 Score", "PR-AUC (Average Precision)"],
    ascending=False
).reset_index(drop=True)

comparison_path = os.path.join(
    RESULT_DIR, "model_comparison.csv"
)

results_df.to_csv(comparison_path, index=False)

print("\nMODEL COMPARISON RESULTS")
print(results_df.to_string(index=False))

print("\nComparison CSV saved:", comparison_path)

# --------------------------------------------------
# 6. SELECT PROVISIONAL BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model_path = trained_models[best_model_name]
best_predictions = model_predictions[best_model_name]

print("\nProvisional best model:", best_model_name)

# Save best model separately
best_model = joblib.load(best_model_path)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_model_provisional.joblib")
)

# Save its classification report
report = classification_report(
    y_test,
    best_predictions,
    target_names=["Genuine", "Fraudulent"],
    zero_division=0
)

report_path = os.path.join(
    RESULT_DIR, "best_model_classification_report.txt"
)

with open(report_path, "w", encoding="utf-8") as file:
    file.write("Provisional Best Model: " + best_model_name)
    file.write("\n\n")
    file.write(report)

print("\nClassification report:")
print(report)

# --------------------------------------------------
# 7. GRAPH 1: MODEL METRIC COMPARISON
# --------------------------------------------------

metrics = [
    "Precision",
    "Recall",
    "F1 Score",
    "PR-AUC (Average Precision)"
]

plot_df = results_df.set_index("Model")[metrics]

ax = plot_df.plot(
    kind="bar",
    figsize=(14, 7),
    width=0.8
)

ax.set_title("Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Models")
ax.set_ylim(0, 1.05)
ax.tick_params(axis="x", rotation=35)
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "model_metrics_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 8. GRAPH 2: ROC CURVES
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for model_name, scores in prediction_scores.items():

    fpr, tpr, _ = roc_curve(y_test, scores)

    auc_value = roc_auc_score(y_test, scores)

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "roc_curve_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 9. GRAPH 3: PRECISION-RECALL CURVES
# --------------------------------------------------

plt.figure(figsize=(9, 7))

for model_name, scores in prediction_scores.items():

    precision_values, recall_values, _ = (
        precision_recall_curve(y_test, scores)
    )

    ap = average_precision_score(y_test, scores)

    plt.plot(
        recall_values,
        precision_values,
        label=f"{model_name} (AP={ap:.3f})"
    )

plt.title("Precision-Recall Curve Comparison")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "precision_recall_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 10. GRAPH 4: BEST MODEL CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(y_test, best_predictions)

plt.figure(figsize=(7, 6))

plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix - " + best_model_name)
plt.colorbar()

class_names = ["Genuine", "Fraudulent"]

plt.xticks([0, 1], class_names)
plt.yticks([0, 1], class_names)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "best_model_confusion_matrix.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 11. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 60)
print("MODEL TRAINING AND COMPARISON COMPLETED")
print("=" * 60)

print("\nSaved models folder:")
print(MODEL_DIR)

print("\nSaved comparison report:")
print(comparison_path)

print("\nSaved graphs folder:")
print(FIGURE_DIR)

print("\nGenerated graphs:")
print("1. model_metrics_comparison.png")
print("2. roc_curve_comparison.png")
print("3. precision_recall_comparison.png")
print("4. best_model_confusion_matrix.png")
