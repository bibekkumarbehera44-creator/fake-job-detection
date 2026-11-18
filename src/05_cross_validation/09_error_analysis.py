
import os
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

DATA_DIR = os.path.join(
    PROJECT_ROOT, "data", "processed"
)

FEATURE_DIR = os.path.join(DATA_DIR, "features")

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

RESULT_DIR = os.path.join(
    PROJECT_ROOT, "results", "error_analysis"
)

os.makedirs(RESULT_DIR, exist_ok=True)

# --------------------------------------------------
# 2. LOAD DATA AND MODEL
# --------------------------------------------------

print("\nLoading model and test data...")

df = pd.read_csv(
    os.path.join(DATA_DIR, "cleaned_jobs.csv")
)

X_test = load_npz(
    os.path.join(FEATURE_DIR, "combined_test.npz")
)

y_test = pd.read_csv(
    os.path.join(FEATURE_DIR, "y_test.csv")
).iloc[:, 0].values

model = joblib.load(
    os.path.join(MODEL_DIR, "linear_svm_ros.joblib")
)

# --------------------------------------------------
# 3. RECREATE TEST INDICES
# --------------------------------------------------

# Recreate the same stratified split to map
# test predictions to their original job postings.

indices = np.arange(len(df))

_, test_indices = train_test_split(
    indices,
    test_size=0.2,
    random_state=42,
    stratify=df["fraudulent"]
)

if len(test_indices) != len(y_test):
    raise ValueError("Test records do not match the dataset split.")

test_df = df.iloc[test_indices].copy().reset_index(drop=True)

# Verify that labels match
if not np.array_equal(
    test_df["fraudulent"].values,
    y_test
):
    raise ValueError(
        "Test labels do not match. Check the original feature extraction split."
    )

# --------------------------------------------------
# 4. GENERATE PREDICTIONS
# --------------------------------------------------

print("\nGenerating test predictions...")

scores = model.decision_function(X_test)

threshold = -0.10383682637001107

predictions = (
    scores >= threshold
).astype(int)

test_df["actual_label"] = y_test
test_df["predicted_label"] = predictions
test_df["decision_score"] = scores

# --------------------------------------------------
# 5. IDENTIFY INCORRECT PREDICTIONS
# --------------------------------------------------

false_positives = test_df[
    (test_df["actual_label"] == 0) &
    (test_df["predicted_label"] == 1)
].copy()

false_negatives = test_df[
    (test_df["actual_label"] == 1) &
    (test_df["predicted_label"] == 0)
].copy()

print("\nERROR ANALYSIS RESULTS")
print("=" * 45)

print("False Positives:", len(false_positives))
print("False Negatives:", len(false_negatives))
print("Total Errors:", len(false_positives) + len(false_negatives))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# --------------------------------------------------
# 6. SAVE ERROR RECORDS
# --------------------------------------------------

errors = pd.concat(
    [
        false_positives.assign(error_type="False Positive"),
        false_negatives.assign(error_type="False Negative")
    ],
    ignore_index=True
)

errors.to_csv(
    os.path.join(RESULT_DIR, "misclassified_jobs.csv"),
    index=False
)

false_positives.to_csv(
    os.path.join(RESULT_DIR, "false_positives.csv"),
    index=False
)

false_negatives.to_csv(
    os.path.join(RESULT_DIR, "false_negatives.csv"),
    index=False
)

# --------------------------------------------------
# 7. DISPLAY SAMPLE ERRORS
# --------------------------------------------------

display_columns = [
    col for col in [
        "title",
        "company_profile",
        "description",
        "requirements",
        "location",
        "employment_type",
        "fraudulent",
        "actual_label",
        "predicted_label",
        "decision_score"
    ]
    if col in errors.columns
]

print("\nSample Misclassified Job Postings:")

if errors.empty:
    print("No misclassified records found.")
else:
    print(
        errors[display_columns].head(5).to_string(index=False)
    )

print("\nError analysis files saved in:")
print(RESULT_DIR)

print("\nERROR ANALYSIS COMPLETED.")
