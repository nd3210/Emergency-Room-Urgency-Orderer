import json
import os
import random
import time
from collections import Counter
 
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    recall_score, classification_report, confusion_matrix,
    accuracy_score, log_loss, f1_score, cohen_kappa_score
)
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# le = LabelEncoder()
# y_encoded = le.fit_transform(y)

# X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify = y_encoded)

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
# sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# xgb_model = XGBClassifier(
#     objective='multi:softprob',
#     num_class=5,
#     eval_metric='mlogloss',
#     random_state=42,
#     n_jobs=1,
#     colsample_bytree=0.7901480892728447,
#     gamma=0.28163778598819184,
#     learning_rate=0.21169966506357696,
#     max_depth=7,
#     min_child_weight=7,
#     n_estimators=380,
#     reg_alpha=0.41038292303562973,
#     reg_lambda=2.0111022770860973,
#     subsample=0.6915192661966489,
# )
# xgb_model.fit(X_train, y_train, sample_weight= sample_weights)
# probs = xgb_model.predict_proba(X_test)


# override = ['cc_cardiacarrest', 'cc_unresponsive', 'cc_strokealert']
# override = [c for c in override if c in X_test.columns]


# def apply_overrides(probs, preds):
#     preds = preds.copy()
#     if override:
#         critical_mask = (probs[override].sum(axis=1) > 0).values
#         preds[critical_mask] = 0
#     return preds


# def predict_with_threshold(probs, threshold):
#     esi1_prob = probs[:, 0]
#     x_pred = xgb_model.predict(X_test)
#     return np.where(esi1_prob >= threshold, 0, x_pred)


# for threshold in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]:
#     preds = predict_with_threshold(probs, threshold)
#     preds = apply_overrides(X_test, preds)
#     esi1_true = (y_test == 0)
#     esi1_recall = (preds[esi1_true] == 0).mean() if esi1_true.sum() > 0 else float('nan')
#     esi1_precision = (y_test[preds == 0] == 0).mean() if (preds == 0).sum() > 0 else float('nan')
#     print(f"threshold={threshold:.2f}  ESI1 recall={esi1_recall:.3f}  "
#           f"ESI1 precision={esi1_precision:.3f}  flagged as ESI1={np.sum(preds == 0)}")

# thresh = 0.30

# final_preds = predict_with_threshold(probs, thresh)
# final_preds = apply_overrides(X_test, final_preds)

# print(f"\n=== Final evaluation (threshold={thresh}, overrides applied) ===")
# print("Accuracy:", accuracy_score(y_test, final_preds))
# print(classification_report(y_test, final_preds, target_names=[str(c) for c in le.classes_]))
# print(confusion_matrix(y_test, final_preds))

# esi_values = np.array([1, 2, 3, 4, 5])
# severity_score = probs @ esi_values
# overridden_mask = (final_preds == 0)
# severity_score[overridden_mask] = 1.0

# queue = pd.DataFrame({
#     'severity_score': severity_score,
#     'predicted_esi': le.inverse_transform(final_preds),
#     'true_esi': le.inverse_transform(y_test),
# })
# queue = queue.sort_values('severity_score')

# print("\n=== Sample of ranked queue (most urgent first) ===")
# print(queue.head(10))



#Neural Network
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

N_ITER = 15
SELECTION_METRIC = 'qwk'
N_CLASSES = 5
OVERRIDE_COLS = ['cc_cardiacarrest', 'cc_unresponsive', 'cc_strokealert']
 
df = pd.read_csv('new_emergency.csv')
print("Data shape:", df.shape)
 
feature_start = df.columns.get_loc('age')
X = df.iloc[:, feature_start:].copy()
y_raw = df['esi']
 
gender_le = LabelEncoder()
X['gender'] = gender_le.fit_transform(X['gender'])
print("encoded 'gender' as:", list(gender_le.classes_))
 
cc_cols = [c for c in X.columns if c.startswith('cc_')]
X['symptom_count'] = X[cc_cols].sum(axis=1)
X['is_infant'] = (X['age'] < 1).astype(float)
X['is_pediatric'] = (X['age'] < 18).astype(float)
X['is_elderly'] = (X['age'] >= 65).astype(float)
print("Engineered features added. Total columns now:", X.shape[1])
 
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)


override_mask = (X[OVERRIDE_COLS].sum(axis=1) > 0).values
print(f"Rows matching red-flag override rule (combined): {override_mask.sum()} "
      f"({override_mask.mean()*100:.2f}% of data)")
if override_mask.sum() > 0:
    true_esi1_among_override = (y_encoded[override_mask] == 0).mean()
    print(f"Of those, fraction actually labeled ESI1 in the data: {true_esi1_among_override:.3f} "
          f"(sanity check on how trustworthy this rule is)")

 
X_temp, X_test, y_temp, y_test, ovr_temp, ovr_test = train_test_split(
    X, y_encoded, override_mask, test_size=0.2, random_state=42, stratify=y_encoded
)
 
X_temp_model = X_temp[~ovr_temp]
y_temp_model = y_temp[~ovr_temp]
 
X_train, X_val, y_train, y_val = train_test_split(
    X_temp_model, y_temp_model, test_size=0.15, random_state=42, stratify=y_temp_model
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train).astype('float32')
X_val = scaler.transform(X_val).astype('float32')
X_test = scaler.transform(X_test).astype('float32')
 
n_features = X_train.shape[1]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print("n_features:", n_features)
 

continuous_cols = ['age', 'symptom_count']
categorical_col_names = [c for c in X.columns if c not in continuous_cols]
categorical_indices = [X.columns.get_loc(c) for c in categorical_col_names]
 
counts = Counter(y_train)
target = int(np.median(list(counts.values())))
over_strategy = {cls: target for cls, cnt in counts.items() if cnt < target}
under_strategy = {cls: target for cls, cnt in counts.items() if cnt > target}
 
steps = []
if over_strategy:
    steps.append(('over', SMOTENC(categorical_features=categorical_indices,
                                       sampling_strategy=over_strategy, random_state=42)))
if under_strategy:
    steps.append(('under', RandomUnderSampler(sampling_strategy=under_strategy, random_state=42)))
 
if steps:
    resample_pipeline = ImbPipeline(steps)
    X_train, y_train = resample_pipeline.fit_resample(X_train, y_train)
 
print("After resampling, train class distribution:",
      dict(zip(*np.unique(y_train, return_counts=True))))
sample_weights_train = np.ones(len(y_train), dtype='float32')
class_weights_tensor = torch.ones(N_CLASSES, dtype=torch.float32)

 
X_train_t = torch.tensor(X_train.astype('float32'))
y_train_t = torch.tensor(y_train, dtype=torch.long)
sw_train_t = torch.tensor(sample_weights_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)
 
train_dataset = TensorDataset(X_train_t, y_train_t, sw_train_t)
 
 
class ConfigurableNet(nn.Module):
    def __init__(self, n_features, hidden_dims, dropout, ordinal, n_classes=5):
        super().__init__()
        layers_list = []
        in_dim = n_features
        for h in hidden_dims:
            layers_list += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.trunk = nn.Sequential(*layers_list)
        self.ordinal = ordinal
        self.n_classes = n_classes
        if ordinal:
            self.shared_out = nn.Linear(in_dim, 1, bias=False)
            self.bias_0 = nn.Parameter(torch.zeros(1))
            self.bias_decrements_raw = nn.Parameter(torch.zeros(n_classes - 2))
        else:
            self.out = nn.Linear(in_dim, n_classes)
 
    def get_ordinal_biases(self):
        decrements = torch.nn.functional.softplus(self.bias_decrements_raw)
        zero = torch.zeros(1, device=decrements.device)
        return self.bias_0 - torch.cat([zero, torch.cumsum(decrements, dim=0)])
 
    def forward(self, x):
        h = self.trunk(x)
        if self.ordinal:
            return self.shared_out(h) + self.get_ordinal_biases()
        return self.out(h)
 
 
def coral_labels(y, n_classes=5):
    thresholds = torch.arange(n_classes - 1, device=y.device).unsqueeze(0)
    return (y.unsqueeze(1) > thresholds).float()
 
 
def coral_probs_to_class_probs(cum_probs, n_classes=5):
    batch = cum_probs.shape[0]
    p = np.zeros((batch, n_classes))
    p[:, 0] = 1 - cum_probs[:, 0]
    for k in range(1, n_classes - 1):
        p[:, k] = cum_probs[:, k - 1] - cum_probs[:, k]
    p[:, n_classes - 1] = cum_probs[:, n_classes - 2]
    p = np.clip(p, 0, None)
    p = p / p.sum(axis=1, keepdims=True)
    return p 
 
search_space = {
    'hidden_dims': [(128, 64), (256, 128), (128, 64, 32), (256, 128, 64), (256, 128, 64, 32), (512, 256, 128), (64, 32)],
    'dropout': [0.2, 0.3, 0.4, 0.5],
    'learning_rate': [0.001, 0.003, 0.005, 0.01],
    'batch_size': [128, 256, 512, 1024],
    'weight_decay': [0, 1e-5, 1e-4],
} 
 
def sample_config():
    return {k: random.choice(v) for k, v in search_space.items()}
 
 
def compute_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'qwk': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
    }
 
 
def train_one_config(config, max_epochs=40, patience=6):
    model = ConfigurableNet(n_features, config['hidden_dims'], config['dropout'],
                             ordinal=True, n_classes=N_CLASSES).to(device)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
 
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
 
    for epoch in range(max_epochs):
        model.train()
        for xb, yb, wb in train_loader:
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            targets = coral_labels(yb, N_CLASSES)
            per_task_loss = bce(logits, targets).mean(dim=1)
            loss = (per_task_loss * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
 
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_tensor.to(device))
            val_targets = coral_labels(y_val_tensor.to(device), N_CLASSES)
            val_loss = nn.functional.binary_cross_entropy_with_logits(val_logits, val_targets).item()
        scheduler.step(val_loss)
 
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
 
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_tensor.to(device))
        cum_probs = torch.sigmoid(val_logits).cpu().numpy()
        val_probs = coral_probs_to_class_probs(cum_probs, N_CLASSES)
        val_preds = np.argmax(val_probs, axis=1)
 
    metrics = compute_metrics(y_val, val_preds)
    return model, best_val_loss, metrics


results = []
best_overall = {'score': -1, 'model': None, 'config': None}
 
search_start = time.time()
for i in range(N_ITER):
    config = sample_config()
    config_start = time.time()
    model, val_loss, metrics = train_one_config(config)
    elapsed = time.time() - config_start
    print(f"[{i+1}/{N_ITER}] {config} -> val_loss={val_loss:.4f}  acc={metrics['accuracy']:.4f}  "
          f"recall_macro={metrics['recall_macro']:.4f}  f1_macro={metrics['f1_macro']:.4f}  "
          f"qwk={metrics['qwk']:.4f}  ({elapsed:.1f}s)")
    results.append({**config, 'val_loss': val_loss, **metrics})
    if metrics[SELECTION_METRIC] > best_overall['score']:
        best_overall = {'score': metrics[SELECTION_METRIC], 'model': model, 'config': config, 'metrics': metrics}
 
total_elapsed = time.time() - search_start
print(f"\nSearch complete in {total_elapsed/60:.1f} minutes")
print(f"Selection metric: {SELECTION_METRIC}")
 
results_df = pd.DataFrame(results).sort_values(SELECTION_METRIC, ascending=False)
print("\n=== Search results, best first ===")
print(results_df.to_string(index=False))
print(f"\nBest config: {best_overall['config']}")
 

best_model = best_overall['model']
best_model.eval()
with torch.no_grad():
    test_logits = best_model(X_test_tensor.to(device))
    cum_probs = torch.sigmoid(test_logits).cpu().numpy()
    test_probs = coral_probs_to_class_probs(cum_probs, N_CLASSES)
model_preds = np.argmax(test_probs, axis=1)
 
final_preds = np.where(ovr_test, 0, model_preds)
 
print("\n=== Model-only test metrics (no rule override applied) ===")
print(compute_metrics(y_test, model_preds))
 
print("\n=== Final: rule + model combined test evaluation ===")
final_metrics = compute_metrics(y_test, final_preds)
print("Accuracy:", final_metrics['accuracy'])
off_by_one_acc = (np.abs(final_preds - y_test) <= 1).mean()
print("Off-by-one accuracy (prediction within 1 ESI level of truth):", off_by_one_acc)
print(classification_report(y_test, final_preds, target_names=[str(c) for c in le.classes_]))
print(confusion_matrix(y_test, final_preds))
 
print("\n=== Threshold sweep for ESI 1 recall (model-only probabilities, before rule override) ===")
for threshold in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.07, 0.05]:
    esi1_prob = test_probs[:, 0]
    argmax_pred = np.argmax(test_probs, axis=1)
    preds = np.where(esi1_prob >= threshold, 0, argmax_pred)
    preds = np.where(ovr_test, 0, preds)  # still apply the rule on top
    esi1_true = (y_test == 0)
    esi1_recall = (preds[esi1_true] == 0).mean() if esi1_true.sum() > 0 else float('nan')
    esi1_precision = (y_test[preds == 0] == 0).mean() if (preds == 0).sum() > 0 else float('nan')
    print(f"threshold={threshold:.2f}  ESI1 recall={esi1_recall:.3f}  "
          f"ESI1 precision={esi1_precision:.3f}  flagged as ESI1={np.sum(preds == 0)}")
 

os.makedirs('model_artifacts', exist_ok=True)
torch.save(best_model.state_dict(), 'model_artifacts/nn_search_best.pt')
with open('model_artifacts/nn_search_best_config.json', 'w') as f:
    json.dump({
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in best_overall['config'].items()},
        'selection_metric': SELECTION_METRIC,
        'override_cols': OVERRIDE_COLS,
        'n_features': n_features,
        'n_classes': N_CLASSES,
    }, f)
joblib.dump(scaler, 'model_artifacts/scaler.joblib')
with open('model_artifacts/feature_columns.json', 'w') as f:
    json.dump(list(X.columns), f)
with open('model_artifacts/label_classes.json', 'w') as f:
    json.dump([float(c) for c in le.classes_], f)
results_df.to_csv('nn_search_results.csv', index=False)
print("\nSaved model + scaler + feature_columns.json + label_classes.json to model_artifacts/")
print("These are what the mobile app's backend server needs to serve predictions.")
print("Saved full search results to nn_search_results.csv")