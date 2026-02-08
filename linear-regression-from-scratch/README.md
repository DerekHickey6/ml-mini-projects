# Linear Regression From Scratch

A clean implementation of linear regression using gradient descent, built from scratch to demonstrate a deep understanding of optimization, loss functions, and model evaluation.

---

## Overview

The model learns a linear relationship


y = wx + b


by minimizing Mean Squared Error (MSE) via gradient descent.  
A synthetic dataset with additive noise is used so the true parameters are known.

---

## Features

- Manual gradient computation  
- Batch gradient descent optimization  
- Loss and parameter history tracking  
- Loss curve visualization (log-scaled)  
- Animated regression line during training  
- Train / test split evaluation  
- Multiple regression metrics  
- Pandas-based evaluation summary  

---

## Dataset

Synthetic linear data:


y = 1.5x + 2 + n


where n is additive noise.  
This allows direct comparison between learned and true parameters.

---

## Evaluation Metrics

Metrics are reported on train and test sets:

- MSE  
- RMSE  
- MAE  
- R²  

Results show strong generalization with train and test errors closely aligned.  
Non-zero error reflects irreducible noise in the data.

---

## Visualizations

- Loss vs epoch (log scale)  
- Animated convergence of the fitted regression line  

These highlight how gradient descent updates parameters over time.

