
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import load_npz, save_npz
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler, SMOTE


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "data" / "processed" / "features"
OUTPUT_DIR = ROOT / "data" / "processed" / "imbalance"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD TRAINING DATA ONLY
# --------------------------------------------------

X_train = load_npz(FEATURE_DIR / "combined_train.npz")

y_train = pd.read_csv(
    FEATURE_DIR / "y_train.csv"
)["fraudulent"]

print("Training data loaded successfully.")
print("Original feature shape:", X_train.shape)

print("\nOriginal class distribution:")
print(y_train.value_counts().sort_index())


# --------------------------------------------------
# 3. CLASS WEIGHTS
# --------------------------------------------------

classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = {
    int(label): float(weight)
    for label, weight in zip(classes, weights)
}

print("\nCLASS WEIGHT METHOD")
print("-------------------")
print("Class weights:", class_weights)

# Save class weights for later model training
joblib.dump(
    class_weights,
    OUTPUT_DIR / "class_weights.joblib"
)


# --------------------------------------------------
# 4. RANDOM OVER-SAMPLING
# --------------------------------------------------

print("\nApplying Random Over-Sampling...")

ros = RandomOverSampler(random_state=42)

X_ros, y_ros = ros.fit_resample(
    X_train,
    y_train
)

save_npz(
    OUTPUT_DIR / "combined_ros_train.npz",
    X_ros
)

pd.DataFrame({
    "fraudulent": y_ros
}).to_csv(
    OUTPUT_DIR / "y_ros_train.csv",
    index=False
)

print("Random Over-Sampling completed.")
print("New feature shape:", X_ros.shape)

print("\nROS class distribution:")
print(pd.Series(y_ros).value_counts().sort_index())


# --------------------------------------------------
# 5. SMOTE
# --------------------------------------------------

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42,
    k_neighbors=5
)

X_smote, y_smote = smote.fit_resample(
    X_train,
    y_train
)

save_npz(
    OUTPUT_DIR / "combined_smote_train.npz",
    X_smote
)

pd.DataFrame({
    "fraudulent": y_smote
}).to_csv(
    OUTPUT_DIR / "y_smote_train.csv",
    index=False
)

print("SMOTE completed.")
print("New feature shape:", X_smote.shape)

print("\nSMOTE class distribution:")
print(pd.Series(y_smote).value_counts().sort_index())


# --------------------------------------------------
# 6. FINAL SUMMARY
# --------------------------------------------------

print("\nIMBALANCE HANDLING COMPLETED!")
print("-----------------------------")

print("Original training records:", len(y_train))
print("ROS training records:", len(y_ros))
print("SMOTE training records:", len(y_smote))

print("\nAll files saved in:")
print(OUTPUT_DIR)
