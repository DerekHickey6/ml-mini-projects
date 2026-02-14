# KNN Classifier From Scratch

A clean implementation of the K-Nearest Neighbors algorithm built from scratch, applied to the Iris dataset to demonstrate understanding of distance-based classification, feature scaling, and evaluation metrics.

## Overview

The classifier predicts flower species by finding the `k` closest training samples (Euclidean distance) and assigning the majority class. A majority-class baseline is included to contextualize KNN performance.

## Features

- KNN classifier with configurable `k`
- Tie-breaking via average neighbor distance
- Majority-class baseline classifier
- Stratified train/test split with reproducible shuffling
- Standard scaler (fit on train only to prevent data leakage)
- Full classification metrics suite
- Modular experiment runner

## Dataset

The [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) contains 150 samples across 3 classes (Setosa, Versicolor, Virginica), each with 4 features: sepal length, sepal width, petal length, and petal width. Classes are balanced at 50 samples each.

## Why Scale?

KNN relies on Euclidean distance, so features with larger ranges dominate the distance calculation. Standard scaling normalizes each feature to zero mean and unit variance, ensuring all features contribute equally. The scaler is fit on training data only — fitting on the full dataset would leak test set statistics into the model.

## Stratified Splitting

The train/test split is stratified — each class is split independently so that both sets preserve the original class distribution. This prevents one class from being underrepresented (or absent) in either set, which is critical for reliable evaluation.

## Metric Choice

All metrics are derived from a confusion matrix, which tracks per-class true positives, false positives, and false negatives. Accuracy alone is misleading when compared against a baseline that always predicts the majority class (33% on a balanced 3-class problem). Macro-averaged precision, recall, and F1 evaluate performance per class and average equally, ensuring no class is ignored.

## Results

| Metric          | Baseline | KNN (k=5) |
|-----------------|----------|------------|
| Accuracy        | 0.333    | 1.000      |
| Macro Precision | 0.111    | 1.000      |
| Macro Recall    | 0.333    | 1.000      |
| Macro F1        | 0.167    | 1.000      |

The baseline predicts all samples as class 0, scoring ~33% accuracy on a balanced dataset. KNN achieves perfect classification, demonstrating that the Iris classes are well-separated in scaled feature space.

## Project Structure

```
iris-classification/
├── scripts/
│   └── experiment_runner.py
├── src/
│   ├── models/
│   │   ├── knn.py
│   │   └── baseline.py
│   ├── preprocessing/
│   │   ├── train_test_split.py
│   │   └── standard_scaler.py
│   ├── metrics/
│   │   └── classification.py
│   └── tests/
│       ├── test_knn.py
│       ├── test_split.py
│       └── test_conf_mat.py
├── README.md
└── requirements.txt
```

## How to Run

```bash
python -m scripts.experiment_runner
```
