# Sentiment Analysis

Binary sentiment classification on IMDB movie reviews using two approaches: a classical TF-IDF pipeline with traditional ML classifiers, and zero-shot inference with a pre-trained DistilBERT model via Hugging Face.

## Overview

Compares classical NLP (TF-IDF vectorization + Logistic Regression / LinearSVC) against a transformer-based approach (DistilBERT) on the same dataset. The TF-IDF models are trained directly on IMDB data. DistilBERT is evaluated zero-shot — fine-tuned on SST-2 (Rotten Tomatoes snippets), not IMDB — making the comparison a measure of out-of-domain transfer performance versus in-domain supervised learning.

## Features

- Modular text preprocessing pipeline in `src/preprocessing.py`
- TF-IDF vectorization with sklearn, fit on training data only to prevent data leakage
- Logistic Regression and LinearSVC classifiers with full evaluation
- Zero-shot DistilBERT inference via Hugging Face `pipeline`
- Confusion matrix comparison across all three models
- Model performance bar chart (accuracy and F1)

## Dataset

The [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) contains 50,000 movie reviews labelled as positive or negative. The dataset is perfectly balanced at 25,000 reviews per class with no missing values. Reviews vary significantly in length — some exceed DistilBERT's 512-token limit and are truncated during inference.

## NLP Pipeline

**Preprocessing** (`src/preprocessing.py`):
- Lowercasing
- HTML tag removal (handles `<br />` tags common in IMDB reviews)
- Punctuation removal
- Digit removal

All operations are applied via pandas `.str` accessor for vectorized performance.

**TF-IDF Vectorization**:
- `TfidfVectorizer` fit on training split only
- Each review represented as a sparse vector of TF-IDF weights
- TF-IDF score = term frequency (normalized by document length) × inverse document frequency (log-scaled)
- High scores indicate words that are frequent in a review but rare across the corpus — strong sentiment signal

**DistilBERT**:
- Model: `distilbert-base-uncased-finetuned-sst-2-english`
- Loaded via Hugging Face `pipeline("sentiment-analysis")`
- Reviews exceeding 512 tokens are truncated
- Evaluated on 1,000-sample subset due to CPU inference time

## Results

| Model | Accuracy | Macro F1 | Notes |
|-------|----------|----------|-------|
| Logistic Regression | 0.89 | 0.89 | Trained on IMDB |
| LinearSVC | 0.90 | 0.90 | Trained on IMDB |
| DistilBERT | 0.86 | 0.86 | Zero-shot (SST-2 fine-tuned) |

LinearSVC performs best among the trained models, consistent with its strength on high-dimensional sparse data. DistilBERT achieves competitive performance despite never being trained on IMDB — fine-tuning on IMDB training data would likely push accuracy to 93–94%.

The DistilBERT model shows a bias toward predicting POSITIVE (recall = 0.96 for class 1, recall = 0.79 for class 0), a known characteristic of SST-2 fine-tuned models when applied to longer, more nuanced reviews.

## Project Structure

```
sentiment-analysis/
├── data/
│   └── IMDB Dataset.csv
├── notebooks/
│   └── sentiment_analysis.ipynb
├── src/
│   └── preprocessing.py
├── visuals/
│   ├── confusion_matrices.png
│   └── model_comparison.png
└── README.md
```

## How to Run

Open the notebook directly in Jupyter:

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

Requires: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `transformers`, `torch`

```bash
pip install transformers torch
```
