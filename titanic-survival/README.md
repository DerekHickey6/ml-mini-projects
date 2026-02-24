# Titanic Survival Prediction

End-to-end classification project on the Kaggle Titanic dataset, demonstrating real-world data cleaning, feature engineering, model comparison, and evaluation.

## Overview

Predicts passenger survival using Logistic Regression and Random Forest. The focus is on handling messy real-world data — missing values, categorical encoding, and engineered features — rather than building models from scratch.

## Features

- Exploratory data analysis with visualizations
- Automated missing value handling (proportional strategy)
- Feature engineering (family size, title extraction, life stage bins, cabin presence)
- One-hot encoding with `drop_first` to avoid multicollinearity
- Logistic Regression and Random Forest comparison
- Evaluation with accuracy, precision, recall, F1, and ROC AUC
- 5-fold stratified cross-validation via Pipeline
- Feature importance visualization

## Dataset

The [Kaggle Titanic dataset](https://www.kaggle.com/competitions/titanic/data) contains 891 passenger records with 12 features including class, sex, age, fare, and embarkation port. The target is binary: survived (1) or died (0). The dataset is imbalanced at ~38% survival rate.

## Preprocessing

Missing values are handled by proportion:
- **>30% missing** (Cabin) — binarized to `HasCabin` and original column dropped
- **<5% missing** (Embarked) — filled with mode
- **5-30% missing** (Age) — filled with median (numeric columns only)

Engineered features:
- **FamilySize** — `SibSp + Parch + 1`
- **IsAlone** — binary flag for solo travelers
- **LifeStage** — age binned into Child, Young Adult, Middle-Aged, Senior
- **Title** — extracted from passenger name (Mr, Mrs, Miss, Master, Other)

## Data Leakage

Feature scaling is performed after the train/test split — the scaler is fit on training data only and applied to both sets. Cross-validation uses a Pipeline to ensure scaling happens inside each fold, preventing test data from influencing the transform.

## Results

| Metric        | Logistic Regression | Random Forest |
|---------------|---------------------|---------------|
| Accuracy      | 0.838               | 0.849         |
| Precision     | 0.770               | 0.778         |
| Recall        | 0.758               | 0.790         |
| F1            | 0.764               | 0.784         |
| ROC AUC       | 0.870               | 0.889         |
| Mean CV Score | 0.833               | 0.810         |

Both models perform similarly. Logistic Regression has a higher cross-validation score, suggesting more stable generalization. Random Forest shows stronger single-split performance but more variance across folds.

## What Features Mattered

- **Sex** — strongest predictor across both models. Being male significantly decreased survival probability.
- **Pclass** — higher ticket class correlated with higher survival. First class passengers had priority access to lifeboats.
- **Fare** — correlated with class and survival. Higher fares reflect better accommodations and proximity to lifeboats.
- **Title** — encodes age, gender, and social status. "Mr" (adult male) was the most negative predictor; "Mrs" and "Miss" were positive.
- **FamilySize / IsAlone** — solo travelers and very large families had lower survival rates.

## Project Structure

```
titanic-survival/
├── data/
│   └── train.csv
├── notebooks/
│   └── eda.ipynb
├── scripts/
│   └── experiment_runner.py
├── src/
│   └── preprocessing.py
├── feature_importance.png
├── README.md
└── requirements.txt
```

## How to Run

```bash
python -m scripts.experiment_runner
```
