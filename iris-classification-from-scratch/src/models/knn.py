# KNN Classifier from scratch
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k_val: int=3):
        if k_val < 1:
            raise ValueError(f"K_val must be positive integer. Got {k_val}")

        self.k_val = k_val
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # Validate X and y
        if not (len(X) and len(y)):
            raise ValueError("Features and labels can not be empty")

        if not (len(X) == len(y)):
            raise ValueError("Shape mismatch: Features and labels must be length")

        if self.k_val > len(X):
            raise ValueError("K Value can not be greater than number of objects")

        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)

        return self

    def predict(self, X):
        X = np.asarray(X)
        # Test cases
        # Allows single sample: (D,) -> (1, D)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Reject invalid input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array (N, D). Got ndim={X.ndim}")

        # Ensure model is fitted
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("fit() must be called before predict()")

        # Ensure same feature dimension as training data
        if X.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                f"Feature mismatch: X has {X.shape[1]} features, "
                f"but X_train has {self.X_train.shape[1]}.")

        # Ensure X is not empty
        if len(X) == 0:
            raise ValueError("X must not be empty")

        preds = []

        for sample in range(len(X)):
            # Distances per sample -> (dist, idx)
            distances = [(np.linalg.norm(X[sample] - Xts), idx) for idx, Xts in enumerate(self.X_train)]

            smallest_dist = []

            # Select and remove nearest neighbor k times
            for _ in range(self.k_val):
                smallest = min(distances)
                distances.remove(smallest)
                smallest_dist.append(smallest)

            # Maps nearest neighbor indices to (label, dist)
            sorted_k_label_dist = [(self.y_train[idx], dist) for dist, idx in smallest_dist]

            # Extracts labels, counts/class and determines max votes
            labels = [lab for lab, _ in sorted_k_label_dist]
            counts = Counter(labels)
            max_votes = max(counts.values())

            # Collect label tied for most votes
            tied_labels = [label for label, c in counts.items() if c == max_votes]

            # If no tie, accept majority label
            if len(tied_labels) == 1:
                preds.append(tied_labels[0])
                continue

            # Computes average distance per tied class
            avg_dist = {}
            for lab in tied_labels:
                dists = [dis for lbl, dis in sorted_k_label_dist if lbl == lab]
                avg_dist[lab] = sum(dists) / len(dists)

            # Picks classes with smallest average neighbor distance
            min_avg = min(avg_dist.values())
            best_labels = [lab for lab, avg in avg_dist.items() if avg == min_avg]

            # Appends minimum label value to preds
            preds.append(min(best_labels))


        return np.asarray(preds, dtype=int)

