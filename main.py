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

xgb_model = XGBClassifier(
    objective='multi:softprob',
    num_class=5,
    eval_metric='mlogloss',
    random_state=42,
    n_jobs=1,
    colsample_bytree=0.7901480892728447,
    gamma=0.28163778598819184,
    learning_rate=0.21169966506357696,
    max_depth=7,
    min_child_weight=7,
    n_estimators=380,
    reg_alpha=0.41038292303562973,
    reg_lambda=2.0111022770860973,
    subsample=0.6915192661966489,
)
xgb_model.fit(X_train, y_train, sample_weight= sample_weights)
probs = xgb_model.predict_proba(X_test)


override = ['cc_cardiacarrest', 'cc_unresponsive', 'cc_strokealert']
override = [c for c in override if c in X_test.columns]


def apply_overrides(probs, preds):
    preds = preds.copy()
    if override:
        critical_mask = (probs[override].sum(axis=1) > 0).values
        preds[critical_mask] = 0
    return preds


def predict_with_threshold(probs, threshold):
    esi1_prob = probs[:, 0]
    x_pred = xgb_model.predict(X_test)
    return np.where(esi1_prob >= threshold, 0, x_pred)


for threshold in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]:
    preds = predict_with_threshold(probs, threshold)
    preds = apply_overrides(X_test, preds)
    esi1_true = (y_test == 0)
    esi1_recall = (preds[esi1_true] == 0).mean() if esi1_true.sum() > 0 else float('nan')
    esi1_precision = (y_test[preds == 0] == 0).mean() if (preds == 0).sum() > 0 else float('nan')
    print(f"threshold={threshold:.2f}  ESI1 recall={esi1_recall:.3f}  "
          f"ESI1 precision={esi1_precision:.3f}  flagged as ESI1={np.sum(preds == 0)}")

thresh = 0.30

final_preds = predict_with_threshold(probs, thresh)
final_preds = apply_overrides(X_test, final_preds)

print(f"\n=== Final evaluation (threshold={thresh}, overrides applied) ===")
print("Accuracy:", accuracy_score(y_test, final_preds))
print(classification_report(y_test, final_preds, target_names=[str(c) for c in le.classes_]))
print(confusion_matrix(y_test, final_preds))

esi_values = np.array([1, 2, 3, 4, 5])
severity_score = probs @ esi_values
overridden_mask = (final_preds == 0)
severity_score[overridden_mask] = 1.0

queue = pd.DataFrame({
    'severity_score': severity_score,
    'predicted_esi': le.inverse_transform(final_preds),
    'true_esi': le.inverse_transform(y_test),
})
queue = queue.sort_values('severity_score')

print("\n=== Sample of ranked queue (most urgent first) ===")
print(queue.head(10))