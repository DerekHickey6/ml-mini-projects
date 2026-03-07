import pandas as pd

def clean_text(df, col='review'):
    df[col] = df[col].str.lower().str.replace(r'<.*?>', '', regex=True).str.replace(r'[^\w\s]', '', regex=True).str.replace(r'[0-9]+', '', regex=True)

    return df