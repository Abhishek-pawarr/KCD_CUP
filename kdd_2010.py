# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import make_classification
from sklearn.model_selection import RandomizedSearchCV

# Load dataset
data = pd.read_csv('C:\\Kdd cup 2010\\train.csv', sep='\t')  # Adjust path as necessary

# Data Prep/Cleaning
data['Step Start Time'] = pd.to_datetime(data['Step Start Time'], format='%Y-%m-%d %H:%M:%S.%f')
data['Step End Time'] = pd.to_datetime(data['Step End Time'], format='%Y-%m-%d %H:%M:%S.%f')
data['Step Duration'] = (data['Step End Time'] - data['Step Start Time']).dt.total_seconds()
data.fillna({'Step Duration': 0}, inplace=True)

# Feature Engineering
data['Time Since Last Step'] = data.groupby('Anon Student Id')['Step Start Time'].diff().dt.total_seconds().fillna(0)
data['Cumulative Hints'] = data.groupby('Anon Student Id')['Hints'].cumsum()
data['Cumulative Correct'] = data.groupby('Anon Student Id')['Corrects'].cumsum()
data['Past Performance'] = data.groupby('Anon Student Id')['Correct First Attempt'].expanding().mean().shift().fillna(0).reset_index(level=0, drop=True)
data['First Attempt Time'] = (pd.to_datetime(data['First Transaction Time']) - data['Step Start Time']).dt.total_seconds().fillna(0)
data['Needed Hint'] = data['Hints'].apply(lambda x: 1 if x > 0 else 0)
data['Incorrect Attempts'] = data['Incorrects']
data['Attempts per Opportunity'] = data.groupby(['Anon Student Id', 'Opportunity(Default)'])['Row'].cumcount() + 1

# Check for missing values
# Fill missing values only in numeric columns
data.fillna(data.select_dtypes(include=[np.number]).mean(), inplace=True)
data['Needed Hint'].fillna(data['Needed Hint'].mode()[0], inplace=True)  # Fill binary column with mode
data.dropna(inplace=True)  # Drop any remaining NA values

# Define feature columns and target variable
selected_features = ['Time Since Last Step', 'Cumulative Hints', 'Past Performance', 'First Attempt Time', 'Needed Hint', 'Incorrect Attempts']
X = data[selected_features]
y = data['Correct First Attempt']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluation
    print(f"Accuracy of {model_name}: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Classification Report of {model_name}:\n{classification_report(y_test, y_pred)}\n")

# Hyperparameter Tuning for Random Forest
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
grid_rf.fit(X_train, y_train)

# Best parameters and score
print(f"Best parameters for Random Forest: {grid_rf.best_params_}")
print(f"Best score for Random Forest: {grid_rf.best_score_}")

# Example SVM Model and Parameter Tuning
param_dist = {
    'C': np.logspace(-3, 3, num=7),
    'gamma': ['scale', 'auto'] + np.logspace(-3, 3, num=7).tolist(),
    'kernel': ['linear', 'rbf']
}

# Setup RandomizedSearchCV for SVM
random_search = RandomizedSearchCV(
    estimator=SVC(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings sampled
    cv=3,       # Cross-validation folds
    verbose=1,
    n_jobs=-1,  # Use all available cores for parallel processing
    random_state=42
)

# Fit the model on the dataset
random_search.fit(X, y)

# Output the best parameters and score for SVM
print(f"Best parameters for SVM: {random_search.best_params_}")
print(f"Best score for SVM: {random_search.best_score_}")

# Gradient Boosting Hyperparameter Tuning
param_grid_gb = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_gb = GridSearchCV(estimator=GradientBoostingClassifier(random_state=42), param_grid=param_grid_gb, cv=3, n_jobs=-1, verbose=2)
grid_gb.fit(X_train, y_train)

# Best parameters and score for Gradient Boosting
print(f"Best parameters for Gradient Boosting: {grid_gb.best_params_}")
print(f"Best score for Gradient Boosting: {grid_gb.best_score_}")

# Evaluation of the best models
best_rf = grid_rf.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print("Random Forest - Best Model Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_rf)}")

best_svm = random_search.best_estimator_
y_pred_svm = best_svm.predict(X_test)
print("SVM - Best Model Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_svm)}")

best_gb = grid_gb.best_estimator_
y_pred_gb = best_gb.predict(X_test)
print("Gradient Boosting - Best Model Performance")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Classification Report:\n{classification_report(y_test, y_pred_gb)}")