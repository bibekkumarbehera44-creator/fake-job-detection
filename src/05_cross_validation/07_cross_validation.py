import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "cleaned_jobs.csv"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "improved_cross_validation"
)

FIGURE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "figures",
    "cross_validation"
)

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD CLEANED DATA
# --------------------------------------------------

print("\nLoading cleaned dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 3. DEFINE COLUMNS
# --------------------------------------------------

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

categorical_columns = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

numeric_columns = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

target_column = "fraudulent"

required_columns = (
    text_columns
    + categorical_columns
    + numeric_columns
    + [target_column]
)

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: " + str(missing_columns)
    )

# Fill text and categorical missing values
for col in text_columns:
    df[col] = df[col].fillna("").astype(str)

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str)

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col], errors="coerce"
    ).fillna(0)

# Create one combined text column
df["combined_text"] = (
    df[text_columns]
    .agg(" ".join, axis=1)
    .str.strip()
)

# --------------------------------------------------
# 4. SEPARATE FEATURES AND TARGET
# --------------------------------------------------

feature_columns = (
    ["combined_text"]
    + categorical_columns
    + numeric_columns
)

X = df[feature_columns].copy()
y = df[target_column].astype(int)

if not set(y.unique()).issubset({0, 1}):
    raise ValueError("Target must contain only 0 and 1.")

print("\nClass distribution:")
print(y.value_counts().sort_index())

# --------------------------------------------------
# 5. RECREATE THE ORIGINAL 80:20 SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Test records:", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

# Test data is deliberately not used in cross-validation.

# --------------------------------------------------
# 6. CREATE PREPROCESSING PIPELINE
# --------------------------------------------------

def create_preprocessor():

    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(
                    max_features=20000,
                    stop_words="english",
                    ngram_range=(1, 2),
                    sublinear_tf=True
                ),
                "combined_text"
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_columns
            ),
            (
                "numeric",
                "passthrough",
                numeric_columns
            )
        ],
        remainder="drop"
    )

# --------------------------------------------------
# 7. DEFINE THREE MODEL PIPELINES
# --------------------------------------------------

models = {

    "Linear SVM - Class Weight": Pipeline([
        ("preprocessing", create_preprocessor()),
        ("classifier", LinearSVC(
            class_weight="balanced",
            max_iter=5000,
            random_state=42
        ))
    ]),

    "Linear SVM - ROS": Pipeline([
        ("preprocessing", create_preprocessor()),
        ("sampler", RandomOverSampler(
            random_state=42
        )),
        ("classifier", LinearSVC(
            max_iter=5000,
            random_state=42
        ))
    ]),

    "Linear SVM - SMOTE": Pipeline([
        ("preprocessing", create_preprocessor()),
        ("sampler", SMOTE(
            random_state=42,
            k_neighbors=5
        )),
        ("classifier", LinearSVC(
            max_iter=5000,
            random_state=42
        ))
    ])
}

# --------------------------------------------------
# 8. STRATIFIED 5-FOLD CROSS-VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision"
}

fold_results = []

summary_results = []

# --------------------------------------------------
# 9. TRAIN AND EVALUATE EACH PIPELINE
# --------------------------------------------------

for model_name, model in models.items():

    print("\n" + "=" * 65)
    print("Running:", model_name)
    print("=" * 65)

    scores = cross_validate(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=1,
        error_score="raise"
    )

    summary_row = {"Model": model_name}

    for metric in scoring:

        values = scores["test_" + metric]

        mean_value = np.mean(values)
        std_value = np.std(values, ddof=1)

        summary_row[metric.upper() + " Mean"] = mean_value
        summary_row[metric.upper() + " Std"] = std_value

        print(
            f"{metric.upper():12s}: "
            f"{mean_value:.4f} +/- {std_value:.4f}"
        )

    summary_results.append(summary_row)

    for fold_number in range(5):

        row = {
            "Model": model_name,
            "Fold": fold_number + 1
        }

        for metric in scoring:
            row[metric.upper()] = scores[
                "test_" + metric
            ][fold_number]

        fold_results.append(row)

# --------------------------------------------------
# 10. SAVE RESULTS
# --------------------------------------------------

summary_df = pd.DataFrame(summary_results)

summary_df = summary_df.sort_values(
    by="F1 Mean",
    ascending=False
).reset_index(drop=True)

fold_df = pd.DataFrame(fold_results)

summary_path = os.path.join(
    RESULT_DIR,
    "improved_cv_summary.csv"
)

fold_path = os.path.join(
    RESULT_DIR,
    "improved_cv_fold_results.csv"
)

summary_df.to_csv(summary_path, index=False)
fold_df.to_csv(fold_path, index=False)

print("\n" + "=" * 65)
print("IMPROVED CROSS-VALIDATION SUMMARY")
print("=" * 65)

print(summary_df.to_string(index=False))

print("\nSummary saved:", summary_path)
print("Fold results saved:", fold_path)

# --------------------------------------------------
# 11. GRAPH 1: F1-SCORE ACROSS FIVE FOLDS
# --------------------------------------------------

plt.figure(figsize=(10, 6))

for model_name in models:

    model_data = fold_df[
        fold_df["Model"] == model_name
    ]

    plt.plot(
        model_data["Fold"],
        model_data["F1"],
        marker="o",
        linewidth=2,
        label=model_name
    )

plt.title("Improved 5-Fold Cross-Validation - F1 Score")
plt.xlabel("Fold")
plt.ylabel("F1 Score")
plt.xticks([1, 2, 3, 4, 5])
plt.ylim(0, 1.05)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "improved_cv_f1_scores.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 12. GRAPH 2: MEAN F1-SCORE
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.bar(
    summary_df["Model"],
    summary_df["F1 Mean"]
)

plt.title("Improved Cross-Validation - Mean F1 Score")
plt.xlabel("Model")
plt.ylabel("Mean F1 Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=20, ha="right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        FIGURE_DIR,
        "improved_cv_mean_f1.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# 13. COMPLETION
# --------------------------------------------------

print("\n" + "=" * 65)
print("IMPROVED CROSS-VALIDATION COMPLETED")
print("=" * 65)

print("\nResults folder:", RESULT_DIR)
print("Graphs folder:", FIGURE_DIR)


# ============================================================
# STEP 8: DECISION THRESHOLD TUNING - TRAINING DATA ONLY
# ============================================================

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

print("\nGenerating training-only out-of-fold decision scores...")

threshold_preprocessor = create_preprocessor()

threshold_pipeline = Pipeline([
    ("preprocessing", threshold_preprocessor),
    ("sampler", RandomOverSampler(random_state=42)),
    ("classifier", LinearSVC(max_iter=5000, random_state=42))
])

threshold_scores = cross_val_predict(
    threshold_pipeline,
    X_train,
    y_train,
    cv=cv,
    method="decision_function",
    n_jobs=1
)

precision, recall, thresholds = precision_recall_curve(
    y_train, threshold_scores
)

f1_values = (
    2 * precision[:-1] * recall[:-1]
    / (precision[:-1] + recall[:-1] + 1e-10)
)

best_index = np.argmax(f1_values)
best_threshold = thresholds[best_index]

print("\n========== BEST THRESHOLD ==========")
print("Best threshold:", best_threshold)
print("Precision:", precision[best_index])
print("Recall:", recall[best_index])
print("F1 Score:", f1_values[best_index])

threshold_results = pd.DataFrame({
    "Threshold": thresholds,
    "Precision": precision[:-1],
    "Recall": recall[:-1],
    "F1 Score": f1_values
})

threshold_dir = os.path.join(PROJECT_ROOT, "results", "cross_validation")
os.makedirs(threshold_dir, exist_ok=True)

threshold_results.to_csv(
    os.path.join(threshold_dir, "threshold_tuning_results.csv"),
    index=False
)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, f1_values)
plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold = {best_threshold:.4f}"
)

plt.xlabel("Decision Threshold")
plt.ylabel("F1 Score")
plt.title("Linear SVM - ROS Decision Threshold Tuning")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(FIGURE_DIR, "threshold_tuning.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close()

print("\nThreshold tuning completed.")
print("Threshold results saved:", threshold_dir)


# ============================================================
# STEP 10: FEATURE IMPORTANCE AND MODEL EXPLAINABILITY
# ============================================================

print("\n" + "=" * 65)
print("STEP 10: FEATURE IMPORTANCE ANALYSIS")
print("=" * 65)

# 1. Load the selected Linear SVM + ROS model

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "linear_svm_ros.joblib"
)

FEATURE_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "features"
)

final_model = joblib.load(MODEL_PATH)

# 2. Load the fitted feature extractors

tfidf = joblib.load(
    os.path.join(FEATURE_DIR, "text_vectorizer.joblib")
)

encoder = joblib.load(
    os.path.join(FEATURE_DIR, "metadata_encoder.joblib")
)

# 3. Extract feature names in the original training order

text_feature_names = [
    "text__" + name
    for name in tfidf.get_feature_names_out()
]

categorical_feature_names = [
    "categorical__" + name
    for name in encoder.get_feature_names_out(categorical_columns)
]

numeric_feature_names = [
    "numeric__" + name
    for name in numeric_columns
]

all_feature_names = (
    text_feature_names
    + categorical_feature_names
    + numeric_feature_names
)

# 4. Extract Linear SVM coefficients

if not hasattr(final_model, "coef_"):
    raise ValueError(
        "The loaded model does not expose Linear SVM coefficients."
    )

coefficients = final_model.coef_.ravel()

# 5. Check whether feature names match model coefficients

if len(all_feature_names) != len(coefficients):
    raise ValueError(
        f"Feature mismatch! Names: {len(all_feature_names)}, "
        f"Coefficients: {len(coefficients)}. "
        "Check the model and feature extraction files."
    )

print("\nTotal features analysed:", len(coefficients))

# 6. Create feature importance table

importance_df = pd.DataFrame({
    "Feature": all_feature_names,
    "Coefficient": coefficients
})

importance_df["Absolute Importance"] = (
    importance_df["Coefficient"].abs()
)

importance_df["Prediction Support"] = np.where(
    importance_df["Coefficient"] > 0,
    "Potential Fake Job",
    "Potential Genuine Job"
)

importance_df = importance_df.sort_values(
    by="Absolute Importance",
    ascending=False
)

# 7. Save feature importance results

IMPORTANCE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "feature_importance"
)

os.makedirs(IMPORTANCE_DIR, exist_ok=True)

importance_path = os.path.join(
    IMPORTANCE_DIR,
    "feature_importance.csv"
)

importance_df.to_csv(importance_path, index=False)

# 8. Select top features for both prediction directions

top_fake = (
    importance_df[importance_df["Coefficient"] > 0]
    .nlargest(15, "Coefficient")
)

top_genuine = (
    importance_df[importance_df["Coefficient"] < 0]
    .nsmallest(15, "Coefficient")
)

plot_df = pd.concat(
    [top_genuine, top_fake]
).sort_values("Coefficient")

# 9. Generate feature importance graph

plt.figure(figsize=(12, 10))

plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"]
)

plt.axvline(0, linewidth=1)
plt.xlabel("Linear SVM Coefficient")
plt.ylabel("Features")
plt.title("Top Features Supporting Fake and Genuine Job Predictions")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()

graph_path = os.path.join(
    IMPORTANCE_DIR,
    "feature_importance.png"
)

plt.savefig(
    graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# 10. Display top results

print("\nTOP FEATURES SUPPORTING FAKE JOB PREDICTIONS:")
print(
    top_fake[["Feature", "Coefficient"]]
    .to_string(index=False)
)

print("\nTOP FEATURES SUPPORTING GENUINE JOB PREDICTIONS:")
print(
    top_genuine[["Feature", "Coefficient"]]
    .to_string(index=False)
)

print("\nFeature importance CSV saved:", importance_path)
print("Feature importance graph saved:", graph_path)

print("\nFEATURE IMPORTANCE ANALYSIS COMPLETED!")
