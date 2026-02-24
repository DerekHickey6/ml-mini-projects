import pandas as pd

from src.preprocessing import clean_df
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score

import matplotlib.pyplot as plt

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


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

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

    # Pipeline
    pipeline = Pipeline([('transformer', scaler), ('classifier', models[i])])

    ## Cross-validation ##
    metrics['Mean CV Score'] = cross_val_score(pipeline, features_df, targets_df, cv=cv, scoring='accuracy').mean()

    conf_mats[i] = confusion_matrix(y_test, y_preds)
    metrics_df[i] = pd.Series(metrics)


## Printing results ##

print(" ##########################")
print(" #   Comparison Metrics   #")
print(" ##########################")
print()
for i in conf_mats:
    print(f"{i} Confusion Matrix")
    print(conf_mats[i])
    print()
print(metrics_df)



## Plot Feature importance & Log-Reg Coefficients##
feature_names = X_train.columns
logreg_importances = models['LogisticRegression'].coef_[0]
rf_importances = models['RandomForestClassifier'].feature_importances_

plt.subplot(1, 2, 1)
plt.barh(feature_names, logreg_importances, label='LogisticRegression')
plt.title('Logistic Regression Coef')
plt.ylabel('Feature')

ax = plt.subplot(1, 2, 2)
plt.barh(feature_names, rf_importances, label='RandomForest')
plt.title('RF Feature Importance')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
ax.set_yticks([])

plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()
