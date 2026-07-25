import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.models import Sequential
# from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import statsmodels.api as sm
import mord
import seaborn as sns

df = pd.read_csv('new_emergency.csv')
print(df.shape)

X = df.iloc[:, df.columns.get_loc('cc_abdominaldistention'): ]
y = df['esi']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify = y_encoded)

#Multinomial Logistic Regression Model
# model = LogisticRegression(solver="saga", max_iter=1000, random_state=42, l1_ratio=1, class_weight='balanced')
# model.fit(X_train, y_train)

# y_pred = model.predict(X_test)
# y_pred_proba = model.predict_proba(X_test)

# test_accuracy = accuracy_score(y_test, y_pred)
# train_accuracy = model.score(X_train, y_train)
# logloss = log_loss(y_test, y_pred_proba)

# print('Training Accuracy', train_accuracy)
# print('Test Accuracy:', test_accuracy, 'Log Loss:', logloss)

# print(classification_report(y_test, y_pred, target_names=[str(c) for c in le.classes_]))
# print(confusion_matrix(y_test, y_pred))

#Ordinal Logistic Regression Model
# model = mord.LogisticAT(alpha=30.0)
# sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# model.fit(X_train, y_train, sample_weight=sample_weights)

# y_pred = model.predict(X_test)

# print("Accuracy:", accuracy_score(y_test, y_pred))
# print(classification_report(y_test, y_pred))

#Decision Tree Model
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
tree_pred = tree_model.predict(X_test)
print(accuracy_score(y_test, tree_pred))
print(mean_squared_error(y_test, tree_pred))

param_grid = {
    'max_depth': [2, 3, 4, 5, 6],
    'min_samples_split': [2, 5, 10, 20],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid_search.fit(X_train, y_train)
print("Best params :", grid_search.best_params_)
print("Best CV acc :", round(grid_search.best_score_, 4))
print("Test acc    :", round(grid_search.best_estimator_.score(X_test, y_test), 4))

