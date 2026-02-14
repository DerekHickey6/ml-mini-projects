# Linear Regression From Scratch

A clean implementation of linear regression using batch gradient descent, built from scratch to demonstrate understanding of optimization, loss functions, and model evaluation.

## Overview

The model learns a linear relationship `y = wx + b` by minimizing Mean Squared Error (MSE) via gradient descent. A synthetic dataset with known true parameters is used, allowing direct comparison between learned and expected values.

## Features

- Manual gradient computation (no autograd)
- Batch gradient descent with configurable learning rate and epochs
- Loss and parameter history tracking
- Train/test split evaluation
- Multiple regression metrics (MSE, RMSE, MAE, R²)
- Pandas-based evaluation summary with generalization analysis
- Loss curve visualization (log-scaled)
- Animated regression line convergence

## Dataset

Synthetic linear data generated from:

`y = 1.5x + 2 + noise`

where noise is drawn from a uniform distribution scaled by a configurable standard deviation. Using synthetic data with known parameters allows direct verification that the model converges to the correct weight and bias.

## Evaluation Metrics

Metrics are reported on both train and test sets, along with the difference between them to assess generalization:

- MSE — average squared error, used as the training loss
- RMSE — square root of MSE, interpretable in the same units as y
- MAE — average absolute error, less sensitive to outliers than MSE
- R² — proportion of variance explained by the model

Train and test errors are closely aligned, confirming the model generalizes well. Non-zero error reflects irreducible noise in the data.

## Visualizations

- Loss vs epoch (log scale) — shows gradient descent converging
- Weight and bias evolution — tracks parameter updates across epochs
- Animated regression line — real-time visualization of the fitted line converging to the data

## Project Structure

```
linear-regression-from-scratch/
├── src/
│   ├── model.py
│   ├── math_utils.py
│   ├── evaluate.py
│   ├── plot_loss.py
│   ├── animation.py
│   └── data/
│       └── datasets.py
├── loss_weight_bias.png
├── README.md
└── requirements.txt
```

## How to Run

```bash
cd src
python evaluate.py
```
