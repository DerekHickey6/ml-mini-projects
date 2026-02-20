import pandas as pd

from src.preprocessing import clean_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

# Load Data
df = pd.read_csv("data/train.csv")

targets_df = df['Survived'].copy()
features_df = clean_df(df)
features_df.drop('Survived', axis=1, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(features_df, targets_df, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression().fit(X_train_scaled, y_train)

y_preds = logreg.predict(X_test_scaled)

print(f"Accuracy:  {accuracy_score(y_test, y_preds):0.3f}")
print(f"F1 Score:  {f1_score(y_test, y_preds):0.3f}")
print(f"Recall:    {recall_score(y_test, y_preds):0.3f}")
print(f"Precision: {precision_score(y_test, y_preds):0.3f}")

print(f"--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_preds))


