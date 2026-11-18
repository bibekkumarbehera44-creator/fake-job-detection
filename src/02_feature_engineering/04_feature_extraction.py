
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scipy.sparse import hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "cleaned_jobs.csv"
OUTPUT_DIR = ROOT / "data" / "processed" / "features"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. COLUMN DEFINITIONS
# --------------------------------------------------

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

CATEGORICAL_COLUMNS = [
    "location",
    "department",
    "salary_range",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function"
]

NUMERIC_COLUMNS = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

TARGET_COLUMN = "fraudulent"


# --------------------------------------------------
# 3. LOAD CLEANED DATASET
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

required_columns = (
    TEXT_COLUMNS
    + CATEGORICAL_COLUMNS
    + NUMERIC_COLUMNS
    + [TARGET_COLUMN]
)

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# --------------------------------------------------
# 4. PREPARE TEXT AND METADATA
# --------------------------------------------------

# Combine all text columns into one text field
text_data = (
    df[TEXT_COLUMNS]
    .fillna("")
    .astype(str)
    .agg(" ".join, axis=1)
)

# Prepare categorical features
categorical_data = (
    df[CATEGORICAL_COLUMNS]
    .fillna("Unknown")
    .astype(str)
)

# Prepare numeric features
numeric_data = (
    df[NUMERIC_COLUMNS]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
)

# Target variable
y = pd.to_numeric(df[TARGET_COLUMN]).astype(int)


# --------------------------------------------------
# 5. TRAIN-TEST SPLIT
# --------------------------------------------------

text_train, text_test, y_train, y_test, cat_train, cat_test, num_train, num_test = (
    train_test_split(
        text_data,
        y,
        categorical_data,
        numeric_data,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

print("\nTrain-test split completed.")
print("Training records:", len(y_train))
print("Testing records:", len(y_test))


# --------------------------------------------------
# 6. TF-IDF TEXT FEATURE EXTRACTION
# --------------------------------------------------

tfidf = TfidfVectorizer(
    max_features=20000,
    stop_words="english",
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_text_train = tfidf.fit_transform(text_train)
X_text_test = tfidf.transform(text_test)

print("\nTF-IDF feature extraction completed.")
print("Text training shape:", X_text_train.shape)
print("Text testing shape:", X_text_test.shape)


# --------------------------------------------------
# 7. ONE-HOT ENCODING
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=True
)

X_cat_train = encoder.fit_transform(cat_train)
X_cat_test = encoder.transform(cat_test)

# Convert numeric metadata into sparse matrices
from scipy.sparse import csr_matrix

X_num_train = csr_matrix(num_train.to_numpy(dtype=float))
X_num_test = csr_matrix(num_test.to_numpy(dtype=float))

# Combine categorical and numeric metadata
X_metadata_train = hstack(
    [X_cat_train, X_num_train],
    format="csr"
)

X_metadata_test = hstack(
    [X_cat_test, X_num_test],
    format="csr"
)

print("\nMetadata encoding completed.")
print("Metadata training shape:", X_metadata_train.shape)
print("Metadata testing shape:", X_metadata_test.shape)


# --------------------------------------------------
# 8. COMBINE TEXT AND METADATA FEATURES
# --------------------------------------------------

X_train = hstack(
    [X_text_train, X_metadata_train],
    format="csr"
)

X_test = hstack(
    [X_text_test, X_metadata_test],
    format="csr"
)

print("\nCombined feature extraction completed.")
print("Combined training shape:", X_train.shape)
print("Combined testing shape:", X_test.shape)


# --------------------------------------------------
# 9. SAVE FEATURE MATRICES
# --------------------------------------------------

save_npz(OUTPUT_DIR / "text_train.npz", X_text_train)
save_npz(OUTPUT_DIR / "text_test.npz", X_text_test)

save_npz(OUTPUT_DIR / "metadata_train.npz", X_metadata_train)
save_npz(OUTPUT_DIR / "metadata_test.npz", X_metadata_test)

save_npz(OUTPUT_DIR / "combined_train.npz", X_train)
save_npz(OUTPUT_DIR / "combined_test.npz", X_test)

# Save target values
pd.DataFrame({"fraudulent": y_train}).to_csv(
    OUTPUT_DIR / "y_train.csv", index=False
)

pd.DataFrame({"fraudulent": y_test}).to_csv(
    OUTPUT_DIR / "y_test.csv", index=False
)

# Save fitted feature extractors
joblib.dump(tfidf, OUTPUT_DIR / "text_vectorizer.joblib")
joblib.dump(encoder, OUTPUT_DIR / "metadata_encoder.joblib")


# --------------------------------------------------
# 10. FINAL SUMMARY
# --------------------------------------------------

print("\nFEATURE EXTRACTION COMPLETED SUCCESSFULLY!")
print("------------------------------------------")
print("Text features:", X_text_train.shape[1])
print("Metadata features:", X_metadata_train.shape[1])
print("Combined features:", X_train.shape[1])

print("\nTraining class distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting class distribution:")
print(y_test.value_counts().sort_index())

print("\nAll feature files saved in:")
print(OUTPUT_DIR)
