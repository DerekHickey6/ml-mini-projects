import pandas as pd

def handle_missing_values(df):
    for column in df.columns:
        null_prop = df[column].isnull().sum() / len(df)

        if null_prop == 0:
            continue

        # Binarizes columns with large amounts of missing data
        if null_prop > 0.3:
            df[f'Has{column}'] = df[column].notnull().astype(int)
            df.drop(column, inplace=True)

        # Fill very small missing data with mode
        elif null_prop < 0.05:
            df[column].fillna(df[column].mode()[0], inplace=True)

        # Else fill missing values with median
        elif null_prop < 0.3 and null_prop > 0.05 and pd.api.types.is_numeric_dtype(df[column]):
            df[column].fillna(df[column].median(), inplace=True)

    return df

def feature_eng(df):
    # Create Family size feature
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    # Create Is Alone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Create age-bins
    df['LifeStage'] = pd.cut(df['Age'], bins=[-1, 10, 30, 60, 100], labels=['Child', 'Young Adult', 'Middle-Aged', 'Senior'])

    # Create title bin
    df['Title'] = df['Name'].str.extract(r'\,\s*(\w+)\.')

    # Labels rare titles as other
    df.loc[~df['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master']), 'Title'] = 'Other'

    # Drop unneeded columns
    df.drop(['Parch', 'SibSp', 'Name', 'Age'], axis=1, inplace=True)

    return df

def encode_categorical(df):
    pass






if __name__ == '__main__':
    df = pd.read_csv("data/train.csv")
    handle_missing_values(df)