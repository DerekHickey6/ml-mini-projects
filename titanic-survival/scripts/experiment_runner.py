import pandas as pd

from src.preprocessing import clean_df
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

models = {}
metrics_df = pd.DataFrame()
conf_mats = {}

# Load Data
df = pd.read_csv("data/train.csv")

targets_df = df['Survived'].copy()
features_df = clean_df(df)
features_df.drop('Survived', axis=1, inplace=True)


X_train, X_test, y_train, y_test = train_test_split(features_df, targets_df, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create model + metrics dictionarys for modular metric processing and display
logreg = LogisticRegression().fit(X_train_scaled, y_train)
models['LogisticRegression'] = logreg

randtree = RandomForestClassifier().fit(X_train_scaled, y_train)
models['RandomForestClassifier'] = randtree


# Calculate metrics
for i in models:
    metrics = {}

    y_preds = models[i].predict(X_test_scaled)
    y_proba = models[i].predict_proba(X_test_scaled)[:, 1]

    metrics['Accuracy'] = accuracy_score(y_test, y_preds)
    metrics['Precision'] = precision_score(y_test, y_preds)
    metrics['Recall'] = recall_score(y_test, y_preds)
    metrics['F1'] = f1_score(y_test, y_preds)

    # Passes prediction scores to roc_auc for better output
    metrics['ROC AUC'] = roc_auc_score(y_test, y_proba)

    conf_mats[i] = confusion_matrix(y_test, y_preds)
    metrics_df[i] = pd.Series(metrics)

## Printing results ##

print(" ##########################")
print(" #   Comparison Metrics   #")
print(" ##########################")
print()
print("Logistic Regression Confusion Matrix")
print(conf_mats['LogisticRegression'])
print("Random Forest Confusion Matrix")
print(conf_mats['RandomForestClassifier'])
print()
print(metrics_df)

