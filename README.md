# Heart Disease Prediction

A machine learning project that predicts the presence of heart disease in patients using K-Nearest Neighbors (KNN) classification on clinical and demographic features.

---

## Overview

This project uses the [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) (`heart.csv`) to build a binary classifier that predicts whether a patient has heart disease (`HeartDisease = 1`) or not (`HeartDisease = 0`). The workflow covers exploratory data analysis, data cleaning, feature selection, model training with hyperparameter tuning, and final evaluation.

---

## Dataset

The dataset contains 918 patient records with the following features:

| Feature | Description |
|---|---|
| `Age` | Age of the patient (years) |
| `Sex` | Sex of the patient (M/F) |
| `ChestPainType` | Type of chest pain (ATA, NAP, ASY, TA) |
| `RestingBP` | Resting blood pressure (mm Hg) |
| `Cholesterol` | Serum cholesterol (mg/dL) |
| `FastingBS` | Fasting blood sugar > 120 mg/dL (1 = true, 0 = false) |
| `RestingECG` | Resting electrocardiogram results |
| `MaxHR` | Maximum heart rate achieved |
| `ExerciseAngina` | Exercise-induced angina (Y/N) |
| `Oldpeak` | ST depression induced by exercise |
| `ST_Slope` | Slope of the peak exercise ST segment (Up, Flat, Down) |
| `HeartDisease` | Target variable (1 = heart disease, 0 = normal) |

---

## Project Structure

```
Heart-Disease-Prediction/
│
├── main.ipynb       # Main Jupyter notebook with full pipeline
├── heart.csv        # Dataset (not included — see Dataset section)
└── README.md        # This file
```

---

## Methodology

### 1. Exploratory Data Analysis
- Distribution plots for all categorical features
- Cross-tab countplots of each categorical feature against the target (`HeartDisease`)

### 2. Data Cleaning
- Removed rows with `RestingBP == 0` (physiologically impossible)
- Imputed `Cholesterol == 0` values with the **median cholesterol**, computed separately for patients with and without heart disease to prevent target leakage

### 3. Feature Engineering
- Applied one-hot encoding (`pd.get_dummies`) to categorical columns with `drop_first=True`
- Computed a correlation heatmap (absolute values) to identify predictive features
- Filtered features with correlation > 0.3 relative to the target

### 4. Feature Selection
The following features were selected for the final model based on univariate KNN experiments and correlation analysis:

- `MaxHR` — Maximum heart rate
- `Oldpeak` — ST depression
- `ExerciseAngina_Y` — Exercise-induced angina (encoded)
- `ST_Slope_Flat` — Flat ST slope (encoded)
- `ST_Slope_Up` — Upward ST slope (encoded)

> **Note:** `Sex_M` was tested but excluded from the final model after inspecting its individual contribution.

### 5. Model Training & Hyperparameter Tuning
- Features scaled using `MinMaxScaler`
- Hyperparameter search via `GridSearchCV` over:
  - `n_neighbors`: 1–19
  - `metric`: `minkowski`, `manhattan`
- Best parameters selected by cross-validated accuracy

### 6. Evaluation
- Final model evaluated on a held-out test set (80/20 split, `random_state=417`)
- Metrics reported: **accuracy score** and **confusion matrix**

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## Usage

1. Clone the repository and place `heart.csv` in the project root
2. Open `main.ipynb` in Jupyter or VS Code
3. Run all cells sequentially

```bash
jupyter notebook main.ipynb
```

---

## Results

The KNN classifier achieved competitive accuracy on the test set after grid search tuning. The confusion matrix revealed the model's performance on both true positive (heart disease detected) and true negative (healthy patient classified correctly) cases.

---

## License

This project is for educational purposes. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and is available under its original license.
