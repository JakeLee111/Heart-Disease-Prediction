# Heart Disease Prediction

A machine learning project that predicts heart disease using k-Nearest Neighbors (k-NN) and XGBoost classifiers, trained on clinical patient data.

## Overview

This project walks through a complete ML pipeline — from raw data to a tuned, evaluated model — using the `heart.csv` dataset (1,061 rows, 12 columns).

## Pipeline

1. **Data Import** — Load and inspect the dataset
2. **Data Cleaning** — Remove duplicates, fix zero-value anomalies in `RestingBP` and `Cholesterol` by replacing them with group medians (split by heart disease status)
3. **EDA** — Bar charts for categorical features; side-by-side comparisons vs. the `HeartDisease` target; correlation heatmap
4. **Preprocessing** — One-hot encoding of categorical columns; feature selection based on correlation threshold (> 0.3); `MinMaxScaler` normalization
5. **Modeling**
   - k-NN trained on individual features, then all selected features combined
   - Hyperparameter tuning via `GridSearchCV` over `n_neighbors` (1–19) and distance metrics (`minkowski`, `manhattan`)
   - XGBoost classifier with feature importance analysis and manual tuning
6. **Evaluation** — Accuracy score, classification report, and confusion matrix

## Selected Features

```
MaxHR, Oldpeak, ExerciseAngina_Y, Sex_M, ST_Slope_Flat, ST_Slope_Up
```

## Tech Stack

| Library | Purpose |
|---|---|
| `pandas` / `numpy` | Data manipulation |
| `matplotlib` / `seaborn` | Visualization |
| `scikit-learn` | k-NN, scaling, grid search, metrics |
| `xgboost` | Gradient boosting classifier |

## Key Findings

### Dataset
- 1,061 patients, 12 features — no missing values, but 143 duplicates removed
- `RestingBP` had 1 zero-value row (dropped); `Cholesterol` had many zeros, replaced with group medians split by heart disease status
- Final cleaned dataset: **917 rows**

### Best Single Feature (k-NN, k=3)
| Feature | Accuracy |
|---|---|
| ST_Slope_Up | 80.98% |
| ST_Slope_Flat | 78.26% |
| ExerciseAngina_Y | 71.20% |
| Oldpeak | 70.65% |
| Sex_M | 55.98% |
| MaxHR | 54.89% |

### Model Comparison (Test Set, 80/20 split)

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| k-NN (tuned, k=18, Minkowski) | 83.70% | 0.84 | 0.84 | 0.84 |
| XGBoost (baseline) | 84.24% | 0.84 | 0.84 | 0.84 |
| XGBoost (tuned) | **88.04%** | **0.88** | **0.88** | **0.88** |

### Takeaways
- `ST_Slope` was the most predictive single feature, outperforming `MaxHR` and `Sex` by a wide margin
- Combining all 6 selected features in k-NN boosted accuracy from ~81% (best single feature) to 83.7%
- XGBoost with tuned hyperparameters (`n_estimators=200`, `learning_rate=0.05`, `max_depth=3`) achieved the best overall result at **88%** accuracy

## Getting Started

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

Place `heart.csv` in the same directory as the notebook, then run all cells in order.

## Dataset

The project expects a CSV file named `heart.csv` with columns including `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`, and `HeartDisease` (target).

A compatible dataset is available on [Kaggle — Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).