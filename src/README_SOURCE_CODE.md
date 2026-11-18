# Organized Source Code — Fake Job Posting Detection

This folder organizes the 22 Python scripts by research task. The original Python code has been kept unchanged; only its folder location has been changed.

## Folder guide

1. `01_data_preparation`
   - `01_load_dataset.py`: download/load the original dataset.
   - `02_data_cleaning.py`: clean missing values and duplicates.
   - `03_eda.py`: explore the dataset and create descriptive analysis.

2. `02_model_training_and_comparison`
   - `04_model_training.py`: baseline model training.
   - `05_feature_experiments.py`: compare text, metadata, and combined features.
   - `06_model_comparison.py`: compare candidate algorithms.
   - `16_final_model_training.py`: inspect before using; confirm it trains the intended final model.

3. `03_explainability_and_interpretation`
   - `07_shap_analysis.py`: SHAP-based interpretation (requires the `shap` package).
   - `08_stability_analysis.py` and `13_explanation_jaccard_stability.py`: explanation stability.
   - `15_fraud_risk_indicators.py`: fraud-associated text indicators.

4. `04_evaluation_and_robustness`
   - `09_error_analysis.py`: inspect false positives and false negatives.
   - `10_imbalance_experiments.py`: compare class-imbalance strategies.
   - `11_cross_validation.py` and `12_model_cross_validation.py`: cross-validation experiments.
   - `14_robustness_testing.py`: perturbation/robustness test.

5. `05_final_model_evaluation`
   - `17`–`20`: ROC, precision-recall, confusion matrix, and metric verification.
   - `21_data_leakage_audit.py`: duplicate and split-overlap audit.
   - `22_group_based_evaluation.py`: exact-duplicate group-based evaluation.

## Suggested research execution order

1. Run dataset loading and cleaning (`01`, then `02`).
2. Run EDA (`03`).
3. Run baseline training, feature experiments, and model comparison (`04`–`06`).
4. Run explainability, imbalance, cross-validation, error, and robustness experiments as needed.
5. Run the leakage audit (`21`) and group-based evaluation (`22`).
6. Generate final figures and verify metrics using the evaluation scripts (`17`–`20`) only after confirming they evaluate the same split and model you report.

## Important research reporting note

The project has both random train/test split results and an exact-duplicate group-based result. Keep them clearly separated in your report. The group-based result previously obtained for Linear SVM was:

- Accuracy: 0.9883
- Precision: 0.9018
- Recall: 0.8497
- F1-score: 0.8750
- ROC-AUC: 0.9907
- PR-AUC: 0.9297

This group-based split had zero overlap of exact `combined_text` groups between train and test. It does not prove that all possible forms of leakage or near-duplicates have been removed.

## Before running scripts

- Open the main project folder in VS Code, not an individual subfolder.
- Check that dataset/model/result paths inside each script still point to the correct project folders.
- Make sure required packages are installed. `07_shap_analysis.py` needs `shap`.
- Some scripts may overwrite output files. Check their save paths before rerunning.
- This package contains only the `src` scripts and this guide. It does not include your dataset, models, app, environment, or results.
