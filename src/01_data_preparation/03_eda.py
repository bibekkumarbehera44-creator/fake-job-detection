
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. SET PROJECT PATHS
# ==========================================

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = os.path.join(
    PROJECT_DIR, "results", "figures"
)

DATASET_PATH = os.path.join(
    PROJECT_DIR, "data", "processed", "cleaned_jobs.csv"
)

os.makedirs(FIGURES_DIR, exist_ok=True)


# ==========================================
# 2. LOAD CLEANED DATASET
# ==========================================

df = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ==========================================
# 3. TARGET VARIABLE DISTRIBUTION
# ==========================================

print("\nFraudulent job distribution:")
print(df["fraudulent"].value_counts())

print("\nFraudulent job percentage:")
print(df["fraudulent"].value_counts(normalize=True) * 100)


plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="fraudulent"
)

plt.title("Distribution of Genuine and Fraudulent Job Postings")
plt.xlabel("Fraudulent (0 = Genuine, 1 = Fraudulent)")
plt.ylabel("Number of Job Postings")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "01_job_distribution.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# 4. COMPANY LOGO ANALYSIS
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="has_company_logo",
    hue="fraudulent"
)

plt.title("Company Logo vs Job Fraud")
plt.xlabel("Company Logo (0 = No, 1 = Yes)")
plt.ylabel("Number of Job Postings")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "02_company_logo_analysis.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# 5. TELECOMMUTING ANALYSIS
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="telecommuting",
    hue="fraudulent"
)

plt.title("Telecommuting vs Job Fraud")
plt.xlabel("Telecommuting (0 = No, 1 = Yes)")
plt.ylabel("Number of Job Postings")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "03_telecommuting_analysis.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# 6. EMPLOYMENT TYPE ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

sns.countplot(
    data=df,
    x="employment_type",
    hue="fraudulent"
)

plt.title("Employment Type vs Job Fraud")
plt.xlabel("Employment Type")
plt.ylabel("Number of Job Postings")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "04_employment_type_analysis.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# 7. JOB DESCRIPTION LENGTH ANALYSIS
# ==========================================

df["description_length"] = (
    df["description"].fillna("").astype(str).str.len()
)

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="fraudulent",
    y="description_length"
)

plt.title("Job Description Length vs Fraud")
plt.xlabel("Fraudulent (0 = Genuine, 1 = Fraudulent)")
plt.ylabel("Description Length (Characters)")

plt.tight_layout()

plt.savefig(
    os.path.join(FIGURES_DIR, "05_description_length_analysis.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==========================================
# 8. COMPLETION MESSAGE
# ==========================================

print("\n==========================================")
print("EDA completed successfully!")
print("All 5 graphs have been saved.")
print("Location:", FIGURES_DIR)
print("==========================================")
