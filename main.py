import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

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
# tree_model = DecisionTreeClassifier(random_state=42, class_weight = 'balanced')
# tree_model.fit(X_train, y_train)
# tree_pred = tree_model.predict(X_test)
# print(accuracy_score(y_test, tree_pred))
# print(mean_squared_error(y_test, tree_pred))

# param_grid = {
#     'max_depth': [4, 6, 8, 10, 15, None],
#     'min_samples_split': [2, 5, 10, 20],
#     'min_samples_leaf': [1, 5, 10],   # add this — helps control overfitting on the leaf side
#     'criterion': ['gini', 'entropy']
# }

# grid_search = GridSearchCV(
#     DecisionTreeClassifier(random_state=42, class_weight='balanced'),
#     param_grid,
#     cv=5,
#     scoring='recall_macro'
# )
# grid_search.fit(X_train, y_train)
# print("Best params :", grid_search.best_params_)
# print("Best CV acc :", round(grid_search.best_score_, 4))
# print("Test acc    :", round(grid_search.best_estimator_.score(X_test, y_test), 4))
# print(classification_report(y_test, grid_search.best_estimator_.predict(X_test)))

#XGBooster
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# xgb_model = XGBClassifier(
#     n_estimators=500,
#     learning_rate=0.1,
#     verbosity=1,
#     random_state=42,
#     num_class=5,
#     max_depth=6,
#     eval_metric='mlogloss',
#     objective='multi:softprob'
# )
# xgb_model.fit(X_train, y_train, sample_weight=sample_weights)

# # make predictions for test data
# y_pred = xgb_model.predict(X_test) 

# # evaluate predictions
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

# importances = pd.Series(xgb_model.feature_importances_, index=X.columns)
# print("\nTop 15 most important features:")
# print(importances.sort_values(ascending=False).head(15))
# print(classification_report(y_test, y_pred))

param_dist = {
    'max_depth': randint(3, 11),              # deeper isn't always better; let search decide
    'learning_rate': uniform(0.01, 0.29),      # 0.01 - 0.30
    'n_estimators': randint(150, 800),
    'subsample': uniform(0.6, 0.4),            # 0.6 - 1.0, row sampling per tree
    'colsample_bytree': uniform(0.6, 0.4),     # 0.6 - 1.0, feature sampling per tree
    'min_child_weight': randint(1, 8),         # higher = more conservative splits
    'gamma': uniform(0, 0.5),                  # minimum loss reduction to split further
    'reg_alpha': uniform(0, 1),                # L1 regularization
    'reg_lambda': uniform(0.5, 2),             # L2 regularization
}
 
base_model = XGBClassifier(
    num_class=5,
    objective='multi:softprob',
    eval_metric='mlogloss',
    random_state=42,
    verbosity=0,
    n_jobs=1
)
 
# scoring='recall_macro' — averages recall equally across all 5 classes,
# which matters far more here than plain accuracy given your class imbalance
random_search = RandomizedSearchCV(
    base_model,
    param_distributions=param_dist,
    n_iter=40,              # 40 random combinations — good coverage without excessive runtime
    scoring='recall_macro',
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=2
)
 
random_search.fit(X_train, y_train, sample_weight=sample_weights)
 
print("\n=== RandomizedSearchCV results ===")
print("Best params:", random_search.best_params_)
print("Best CV recall_macro:", round(random_search.best_score_, 4))
 
best_xgb = random_search.best_estimator_
tuned_pred = best_xgb.predict(X_test)
tuned_proba = best_xgb.predict_proba(X_test)
 
print("Test Accuracy:", accuracy_score(y_test, tuned_pred))
print("Test Log Loss:", log_loss(y_test, tuned_proba))
print(classification_report(y_test, tuned_pred, target_names=[str(c) for c in le.classes_]))
print(confusion_matrix(y_test, tuned_pred))
 
tuned_importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
print("\nTop 15 most important features (tuned model):")
print(tuned_importances.sort_values(ascending=False).head(15))