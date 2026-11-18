
import os
import joblib

files = {
    "Final Model": "models/combined_linear_svm_final.joblib",
    "TF-IDF Vectorizer": "data/processed/features/text_vectorizer.joblib",
    "Metadata Encoder": "data/processed/features/metadata_encoder.joblib"
}

print("\n========== ARTIFACT VERIFICATION ==========\n")

artifacts = {}

for name, path in files.items():

    if not os.path.exists(path):
        print(f"MISSING: {name} -> {path}")
        continue

    obj = joblib.load(path)
    artifacts[name] = obj

    print(f"FOUND: {name}")
    print("Object type:", type(obj).__name__)

    if hasattr(obj, "n_features_in_"):
        print("Input features:", obj.n_features_in_)

    if hasattr(obj, "coef_"):
        print("Coefficient shape:", obj.coef_.shape)

    if hasattr(obj, "vocabulary_"):
        print("TF-IDF vocabulary size:", len(obj.vocabulary_))

    if hasattr(obj, "categories_"):
        print(
            "Encoder categories:",
            len(obj.categories_)
        )

    print("-" * 45)

print("\n========== VERIFICATION COMPLETED ==========")
