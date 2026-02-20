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

# Create model + metrics dictionarys for modular metric processing and display
model_preds = {"logisticRegression": y_preds}
metrics_df = pd.DataFrame()

# Calculate metrics
for i in model_preds:
    metrics = {}

    metrics['Accuracy'] = accuracy_score(y_test, model_preds[i])
    metrics['Precision'] = precision_score(y_test, model_preds[i])
    metrics['Recall'] = recall_score(y_test, model_preds[i])
    metrics['F1'] = f1_score(y_test, model_preds[i])
    metrics['ROC AUC'] = roc_auc_score(y_test, model_preds[i])

    metrics_df[i] = pd.Series(metrics)

logreg_conf_mat = confusion_matrix(y_test, y_preds)

## Printing results ##

print(" ##########################")
print(" #   Comparison Metrics   #")
print(" ##########################")
print()
print("Logistic Regression Confusion Matrix")
print(logreg_conf_mat)
print()
print(metrics_df)

