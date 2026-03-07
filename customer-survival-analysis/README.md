# Customer Survival Analysis

Survival analysis on telecom subscriber data to model customer churn over time, identify high-risk customer profiles, and quantify the impact of contract type, payment method, and service add-ons on retention.

## Overview

Goes beyond binary churn classification by applying survival analysis techniques to answer not just *if* a customer will churn, but *when* — and with what probability they remain a customer at any given point in time. Uses Kaplan-Meier estimation for group-level survival curves and a Cox Proportional Hazards model to quantify the effect of individual features on churn risk.

## Features

- Modular preprocessing pipeline with binary encoding and one-hot encoding
- Kaplan-Meier survival curve for the overall customer population
- Kaplan-Meier comparison by contract type (month-to-month, one year, two year)
- Cox Proportional Hazards model with regularization (penalizer=0.1)
- Hazard ratio forest plot showing feature-level churn risk
- Predicted survival curves for contrasting high-risk and low-risk customer profiles

## Dataset

The [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) contains 7,043 telecom subscribers with 20 features including contract type, payment method, internet service type, and monthly charges. The churn rate is ~26%. Customers still active at the end of the observation period are treated as censored observations.

## Survival Analysis Approach

Standard binary classification treats churned and non-churned customers symmetrically, discarding time information and mishandling censored data. This project uses survival analysis to correctly handle both:

- **Censored observations** — customers still active when data collection ends. Their partial information (they survived at least N months) is used rather than discarded or mislabeled.
- **Time-to-event modeling** — the output is a survival probability curve over time, not a single yes/no prediction.

## Key Findings

- **Contract type is the dominant retention factor** — two year contract customers have a 69% lower churn hazard than month-to-month customers (HR = 0.31). By month 72, only ~15% of month-to-month customers remain active versus ~95% of two year contract customers.
- **Electronic check payment is the biggest churn risk** — customers paying by electronic check churn 56% faster than the baseline (HR = 1.56), likely reflecting disengaged or at-risk customers.
- **Service add-ons significantly reduce churn** — customers with online security (HR = 0.58) and online backup (HR = 0.61) are substantially less likely to churn, suggesting engaged customers who invest in their subscription are more loyal.
- **Fiber optic internet increases churn risk** (HR = 1.38) — likely due to higher monthly costs and greater competition in the fiber market.
- **The Cox model achieves a concordance of 0.86** — correctly ranking churn risk 86% of the time.

## Preprocessing

Handled in `src/preprocessing.py`:
- Rows with blank `TotalCharges` dropped (11 rows, all with `tenure = 0`)
- `TotalCharges` converted from object to float
- `Churn` mapped from Yes/No to 1/0
- Binary Yes/No columns automatically detected and mapped to 1/0
- `gender` mapped to 1/0 (Male/Female)
- Multi-category columns one-hot encoded with `drop_first=True`
- `TotalCharges` dropped before Cox fitting due to near-perfect collinearity with `tenure * MonthlyCharges`

## Results

| Metric | Value |
|--------|-------|
| Concordance (C-statistic) | 0.86 |
| Partial log-likelihood | -14338.49 |
| Log-likelihood ratio test | 2629.11 on 28 df |
| Number of observations | 7032 |
| Number of churn events | 1869 |

## Project Structure

```
customer-survival-analysis/
├── data/
│   └── Telco-Customer-Churn.csv
├── notebooks/
│   └── survival_analysis.ipynb
├── src/
│   └── preprocessing.py
├── visuals/
└── README.md
```

## How to Run

Open the notebook directly in Jupyter:

```bash
jupyter notebook notebooks/survival_analysis.ipynb
```

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `lifelines`

```bash
pip install lifelines
```
