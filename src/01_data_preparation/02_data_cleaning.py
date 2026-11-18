
from pathlib import Path
import pandas as pd

# Find the main project folder
ROOT = Path(__file__).resolve().parents[2]

# Input and output file locations
SOURCE = ROOT / "dataset" / "cleaned_job_postings.csv"
OUTPUT = ROOT / "data" / "processed" / "cleaned_jobs.csv"

# Text columns
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

# Categorical columns
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


def main():

    # Check whether the dataset exists
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {SOURCE}\n"
            "Please check the dataset folder and filename."
        )

    # Load dataset
    df = pd.read_csv(SOURCE)

    print("Original dataset shape:", df.shape)

    # Check required target column
    if "fraudulent" not in df.columns:
        raise ValueError("The 'fraudulent' column is missing.")

    # Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates().copy()

    print("Duplicate rows removed:", duplicate_count)

    # Handle missing text values
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Handle missing categorical values
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    # Convert target column into numeric format
    df["fraudulent"] = pd.to_numeric(
        df["fraudulent"], errors="coerce"
    )

    # Remove rows with missing or invalid target values
    df = df.dropna(subset=["fraudulent"]).copy()

    # Check that target values are only 0 and 1
    if not set(df["fraudulent"].unique()).issubset({0, 1}):
        raise ValueError(
            "The target column must contain only 0 (Genuine) "
            "and 1 (Fraudulent)."
        )

    df["fraudulent"] = df["fraudulent"].astype(int)

    # Remove job ID because it is only an identifier
    if "job_id" in df.columns:
        df = df.drop(columns=["job_id"])

    # Create output folder if it does not exist
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Save cleaned dataset
    df.to_csv(OUTPUT, index=False)

    # Display final results
    print("\nDATA CLEANING COMPLETED")
    print("-----------------------")
    print("Cleaned dataset shape:", df.shape)
    print("Saved file:", OUTPUT)

    print("\nMissing values after cleaning:")
    print(df.isnull().sum().sum())

    print("\nClass distribution:")
    print("0 = Genuine, 1 = Fraudulent")
    print(df["fraudulent"].value_counts().sort_index())

    print("\nClass percentages:")
    print(
        (df["fraudulent"].value_counts(normalize=True) * 100)
        .sort_index()
        .round(2)
    )


if __name__ == "__main__":
    main()
