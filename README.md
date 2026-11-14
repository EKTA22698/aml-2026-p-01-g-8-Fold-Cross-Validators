# Applied Machine Learning Project

## P01: Hourly Bike-Rental Demand for Fleet Rebalancing

**Team Name:** 8 Fold Cross Validators
**Course:** Applied Machine Learning
**Project Code:** P01
**Institution:** UPES, Dehradun

---

## 1. Team Members

| S. No. | Name | SAP ID |
|---|---|---|
| 1 | Ekta Bhardwaj | 590022698 |
| 2 | Adhya Agarwal | 590025832 |
| 3 | Arnav Kansal | 590023355 |
| 4 | Shresth Bansal | 590022218 |
| 5 | Shaurya Sood | 590024336 |
| 6 | Ishanvi Kalra | 590025196 |
| 7 | Aryan Agarwal | 590027693 |
| 8 | Manasvi Maheshwari | 590025321 |

## 2. Project Overview

Bike-sharing services need to maintain an adequate number of bicycles at different rental stations. Demand can change throughout the day depending on factors such as hour, weekday, season and weather.

This project focuses on analysing hourly bike-rental demand and developing machine learning models to estimate the number of rentals. The objective is to understand demand patterns and explore how demand predictions can support better fleet planning and rebalancing decisions.

## 3. Problem Statement

Uneven bike-rental demand can lead to some stations having insufficient bicycles while others have unused bicycles.

The project investigates how historical rental data and time-related and weather-related features can be used to predict hourly rental demand.

## 4. Project Objectives

- Analyse hourly bike-rental demand patterns.
- Understand the relationship between rental demand and time or weather conditions.
- Perform data validation and preprocessing.
- Apply feature engineering techniques.
- Train and compare Ridge and Lasso regression models.
- Evaluate model performance using appropriate regression metrics.
- Explore how demand predictions can support fleet rebalancing.

## 5. Dataset Description

The project uses the UCI Bike Sharing Dataset.

**Dataset files:**
- `hour.csv` – hourly bike-rental records.
- `day.csv` – daily bike-rental records.

The hourly dataset is used for model development.

| Property | Description |
|---|---|
| Dataset | UCI Bike Sharing |
| File used | `hour.csv` |
| Rows | 17,379 |
| Original columns | 17 |
| Target variable | `cnt` |
| Prediction task | Hourly rental count estimation |

The target variable `cnt` represents the total number of bike rentals.

## 6. Data Validation and Preprocessing

The hourly dataset was checked for data quality before model training.

Validation included:
- Checking expected columns and dataset dimensions.
- Checking missing and duplicate records.
- Checking feature ranges and categorical values.
- Verifying that total rentals equal casual plus registered rentals.
- Identifying missing hourly timestamps.

The validation identified 165 missing hourly timestamps across 75 gaps. These missing time periods were not filled with artificial records because their corresponding weather and other feature values were unknown.

The variables `casual` and `registered` were excluded from model inputs because they directly contribute to the target variable `cnt`, which would cause target leakage.

## 7. Feature Engineering

Feature engineering was performed to make time-related patterns more useful for machine learning.

Cyclic encoding was applied to:
- Hour
- Month
- Weekday

Sine and cosine transformations represent the cyclical nature of these variables. For example, hour 23 and hour 0 are close in time, even though their numerical values appear far apart.

A preprocessing pipeline was also used to prepare the input features for model training.

## 8. Machine Learning Models

Two regularized linear regression algorithms were trained and compared.

### Ridge Regression

Ridge regression uses L2 regularization to reduce the effect of large model coefficients and help control overfitting.

### Lasso Regression

Lasso regression uses L1 regularization and can reduce less useful feature coefficients towards zero.

### Training and Evaluation Strategy

- Chronological training and testing split.
- 80% training data and 20% testing data.
- 5-fold time-series cross-validation on the training data.
- Model selection based on cross-validation RMSE.

## 9. Model Evaluation

The models were evaluated using:

- **MAE (Mean Absolute Error):** Average absolute difference between actual and predicted rental counts.
- **RMSE (Root Mean Squared Error):** Penalises larger prediction errors more strongly.
- **R² Score:** Measures the proportion of target variance explained by the model.

### Test Set Results

| Metric | Ridge Regression | Lasso Regression |
|---|---:|---:|
| MAE | 120.7392 | 120.7384 |
| RMSE | 164.4133 | 164.4102 |
| R² Score | 0.4439 | 0.4440 |

Lasso Regression achieved a marginally lower test MAE and RMSE and a slightly higher R² score.

The Lasso model was also selected based on cross-validation RMSE. The difference between the two models is very small.

**Interpretation:** The current linear models provide a baseline for hourly demand prediction. Their performance indicates that further feature development and model experimentation may improve predictions.

## 10. Project Outputs

The `models/` directory contains the generated model and evaluation outputs.

| File | Purpose |
|---|---|
| `best_model.joblib` | Selected trained model |
| `ridge_regression.joblib` | Trained Ridge model |
| `lasso_regression.joblib` | Trained Lasso model |
| `cv_summary.csv` | Cross-validation summary |
| `cv_fold_results.csv` | Individual fold results |
| `test_results.csv` | Test-set evaluation metrics |
| `actual_vs_predicted.png` | Actual versus predicted values |
| `residuals.png` | Residual analysis plot |

## 11. Project Structure

```text
aml-2026-p01-g1/
│
├── data/
│   └── raw/
│       ├── day.csv
│       └── hour.csv
│
├── models/
│   ├── actual_vs_predicted.png
│   ├── best_model.joblib
│   ├── cv_fold_results.csv
│   ├── cv_summary.csv
│   ├── lasso_regression.joblib
│   ├── residuals.png
│   ├── ridge_regression.joblib
│   └── test_results.csv
│
├── notebooks/
│   ├── 01-eda.ipynb
│   └── 02-ridge-lasso-models.ipynb
│
├── src/
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── test_feature_engineering.py
│   └── validation.py
│
├── .gitignore
└── README.md
```

## 12. How to Run the Project

### Requirements

- Python 3.12
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib
- Pytest

### Setup

Clone the repository:

```bash
git clone https://github.com/EKTA22698/aml-2026-p-01-g-8-Fold-Cross-Validators.git
cd aml-2026-p-01-g-8-Fold-Cross-Validators
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install numpy pandas scikit-learn matplotlib joblib pytest
```

Run model training:

```bash
python src/model_training.py
```

Run automated tests:

```bash
python -m pytest -v
```

## 13. Limitations and Future Scope

The current implementation uses historical UCI bike-sharing data from 2011–2012. Its patterns may not represent present-day bike-sharing demand.

The current Ridge and Lasso models are baseline approaches. Further work may include:

- Experimenting with additional regression algorithms.
- Improving feature engineering.
- Analysing station-level demand where suitable data is available.
- Exploring demand forecasting with more recent datasets.
- Developing a fleet rebalancing strategy based on predicted demand.

## 14. Conclusion

This project demonstrates a machine learning workflow for hourly bike-rental demand estimation, including data validation, preprocessing, feature engineering, model training and evaluation.

Ridge and Lasso regression were implemented and evaluated using time-series cross-validation and a chronological test split. Lasso achieved marginally better evaluation results, although both models performed similarly.

The work provides a baseline for understanding bike-rental demand and exploring data-driven fleet rebalancing.

---

**Team 8 Fold Cross Validators**
Applied Machine Learning | UPES Dehradun
