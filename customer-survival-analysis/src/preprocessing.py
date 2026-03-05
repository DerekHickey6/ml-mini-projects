import numpy as np
import pandas as pd

# map columns [gender, parent, dependents, phone service, paperlessBilling] to 1/0
def encode_binary_columns(df):
    for col in df.columns:
        if set(df[col].unique()) <= {'Yes', 'No'}:
            df[col] = df[col].map({'Yes' : 1, 'No': 0})

        if set(df[col].unique()) <= {'Male', 'Female'}:
            df[col] = df[col].map({'Male': 1, 'Female': 0})

    return df

# One-hot encode multi-category columns
# multipleLines, InternetService, onlinesecurity, onlineBackup, deviceprotection, techsupport, streamingTV, StreamingMovies, Contract, PaymentMethod
def encode_categorical_columns(df):
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    return df

# Convenience method to call both on a dataframe
def encode_columns(df):
    df = encode_categorical_columns(encode_binary_columns(df))

    return df