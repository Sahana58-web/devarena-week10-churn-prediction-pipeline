# Customer Churn Prediction: Preprocessing and Feature Engineering

Week 10 - Data Preprocessing and Feature Engineering

Prepares customer data, engineers new features, selects the useful ones and builds a complete preprocessing pipeline that predicts which customers might leave.

## What is done
- **Encoding (5 methods):** label, ordinal, one-hot, binary, frequency
- **Scaling (3 methods):** Min-Max, Standard, Robust (fitted on training data only)
- **Outliers:** IQR and Z-score detection, tested with fake outliers, capping inside the pipeline
- **Feature engineering (9 new features):** TenureGroup_code, FirstYear, EstimatedCLV, IsMonthToMonth, IsAutoPay, HighCharge, Tenure_log, NewMonthToMonth, BillingRatio
- **Feature selection (4 methods):** correlation analysis, F-test, mutual information, Random Forest importance
- **Pipeline:** raw columns -> build features -> cap outliers -> scale -> model, with 5 end-to-end tests

## Key results
| Setup | CV ROC-AUC |
|---|---|
| Baseline (raw numbers) | 0.970 |
| Selected features (include Tenure) + Logistic Regression | 0.994 |
| Sign-up-only features + Logistic Regression | 0.707 |

Scores with Tenure-based features are probably too optimistic, because Tenure may reveal the outcome. The sign-up-only score is the fair estimate. Details in `preprocessing_report.md` and `feature_engineering_documentation.md`.

## Files
- `churn_prediction_pipeline.ipynb` - the notebook
- `churn_data.csv` - the data (500 customers)
- `preprocessing_report.md` - every preprocessing step with its reason, results and tests
- `feature_engineering_documentation.md` - the new features, validation and selection
- `requirements.txt` - Python packages
- `churn_overview.png`, `scaling_comparison.png`, `outlier_boxplots.png`, `new_features.png`, `feature_correlation.png`, `feature_selection.png`, `pipeline_results.png` - charts

## Setup
```bash
pip install -r requirements.txt
```
Open the notebook in Google Colab, upload `churn_data.csv` when asked, and run all cells from top to bottom.
