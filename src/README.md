# Cleaned source files — Fake Job Posting Detection

This is a trimmed source folder for the **current combined-feature project**.
The original archive contained 22 scripts, many from an older text-only
workflow. Those scripts referred to old paths and artifact names such as
`dataset/cleaned_job_postings.csv`, `final_linear_svm.pkl`, and
`final_tfidf_vectorizer.pkl`. They do not match the current project files.

## Essential scripts retained

| File | Purpose |
|---|---|
| `check_project_files.py` | Checks that the current dataset, encoders, test features, labels, and final model exist. |
| `audit_dataset.py` | Checks duplicates, conflicting duplicate labels, and missing target values. |
| `evaluate_final_model.py` | Evaluates the saved combined-feature Linear SVM on the existing held-out test matrix and saves verified metrics. |

## Expected project structure

```text
project-root/
├── src/
│   ├── README.md
│   ├── check_project_files.py
│   ├── audit_dataset.py
│   └── evaluate_final_model.py
├── data/
│   └── processed/
│       ├── cleaned_jobs.csv
│       ├── y_test.csv
│       └── features/
│           ├── text_vectorizer.joblib
│           ├── metadata_encoder.joblib
│           └── combined_test.npz
├── models/
│   └── combined_linear_svm_final.joblib
└── result/
```

## How to run

Open the **project root** in VS Code, then run:

```bash
python src/check_project_files.py
python src/audit_dataset.py
python src/evaluate_final_model.py
```

Install the required packages in your project's virtual environment if needed:

```bash
pip install pandas scipy scikit-learn joblib
```

## Company authenticity verification layer

The project now includes an independent evidence-based verification module at
`src/company_verification.py`. It is designed to complement the existing Linear
SVM fraud detector without retraining or replacing it.

### Risk scoring weights

The verification engine uses the following configurable weights. Total weight is
100, and each category contributes only once per risk dimension to avoid
double-counting correlated indicators.

- Company identity consistency: 18
- Website authenticity: 18
- Email authenticity: 12
- Registration verification: 14
- Contact information consistency: 12
- Recruitment behaviour: 14
- Suspicious payment requests: 8
- Known impersonation or scam reports: 12

### Classification rules

The final assessment combines ML predictions with evidence-based company checks:

- `GENUINE-LOOKING / LOW RISK`: low risk score, adequate evidence, and no serious inconsistency.
- `SUSPICIOUS / NEEDS VERIFICATION`: moderate risk or an unresolved discrepancy that requires manual verification.
- `HIGH RISK / POTENTIAL FRAUD`: strong suspicious indicators, fee requests, impersonation cues, or a high-risk score.
- `UNABLE TO VERIFY`: missing evidence, unavailable external checks, or insufficient independent proof to confirm authenticity.

Important: a company passing basic checks is not automatically treated as genuine.
If the evidence remains weak or unverified, the system returns an uncertainty
status instead of declaring it legitimate.

### External API support

Optional external validation can be enabled through environment variables such as
`MCA_API_KEY`, `BUSINESS_REGISTRY_API_KEY`, or `ENABLE_EXTERNAL_COMPANY_CHECKS`.
When services are unavailable, the system falls back to `unavailable` or
`inconclusive` results instead of fabricating evidence.

## Important

- These scripts assume the artifact names and folder structure listed above.
- The evaluation script uses the already-saved test features and labels. It does
  not retrain the model or recreate the train/test split.
- Keep your existing preprocessing and model-training scripts if you need to
  reproduce the combined TF-IDF + metadata training pipeline. They were not
  reconstructed from the old archive because its feature-generation logic
  does not match the current model artifacts.
- Do not report new metric values until `evaluate_final_model.py` runs
  successfully against the exact saved test set used for your final model.
- Company verification is intentionally conservative. No automated check can
  guarantee that a perfectly disguised fraud scheme is genuine without reliable
  independent evidence.
