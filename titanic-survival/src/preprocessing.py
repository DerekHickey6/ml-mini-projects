import pandas as pd

def handle_missing_values(df):
    for column in df.columns:
        null_prop = df[column].isnull().sum() / len(df)

        if null_prop == 0:
            continue

        # Binarizes columns with large amounts of missing data
        if null_prop > 0.3:
            df[f'Has{column}'] = df[column].notnull().astype(int)
            df.drop(column, axis=1, inplace=True)

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
    df.drop(['Parch', 'SibSp', 'Name', 'Age', 'PassengerId', 'Ticket'], axis=1, inplace=True)

    return df

# One-hot encode remaining attributes
def encode_categorical(df):

    return pd.get_dummies(df,
                          columns=['Sex', 'Embarked', 'LifeStage', 'Title'],
                          prefix=['Sex', 'Embarked', 'LifeStage', 'Title'],
                          dtype=int,
                          drop_first=True)

# Convenience method to call preprocessing steps
def clean_df(df):
    return encode_categorical(feature_eng(handle_missing_values(df)))


if __name__ == '__main__':
    df = pd.read_csv("data/train.csv")
    df = handle_missing_values(df)
    df = feature_eng(df)
    df = encode_categorical(df)
    print(df)