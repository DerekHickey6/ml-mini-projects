# Majority class Baseline
from collections import Counter
import numpy as np

class MajorityClassClassifier:
    def __init__(self):
        self.max_freq_label = None

    # Counts labels and uses lowest value feature in ties
    def fit(self, X, y):
        if len(y) == 0:
            raise ValueError("Labels list can not be empty")

        label_counts = Counter(y)
        max_count = label_counts.most_common(1)[0][1]
        tied_labels = [label for label, count in label_counts.items() if count == max_count]

        self.max_freq_label = min(tied_labels)

        return self

    # Return np.array with shape (len(X),) and filled with highest freq label
    def predict(self, X):
        if self.max_freq_label == None:
            raise RuntimeError("Must call fit() before predict()")

        return np.full((len(X),), self.max_freq_label, dtype=int)

