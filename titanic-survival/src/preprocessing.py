import pandas as pd

def handle_missing_values(df):
    for column in df.columns:
        null_prop = df[column].isnull().sum() / len(df)

        if null_prop == 0:
            continue

        # Binarizes columns with large amounts of missing data
        if null_prop > 0.3:
            df[f'Has{column}'] = df[column].notnull()
            df.drop(column)

        # Fill very small missing data with mode
        if null_prop < 0.05:
            df[column].fillna(df[column].mode(), inplace=True)

        # Else fill missing values with median
        else:
            df[column].fillna(df[column].median(), inplace=True)



df = pd.read_csv("data/train.csv")
handle_missing_values(df)