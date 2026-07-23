import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('new_emergency.csv')
print(df.shape)

X = df.iloc[:, 7: ]
y = df['esi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# reg = linear_model.LogisticRegression(max_iter=10000, random_state=0)
# reg.fit(X_train, y_train)

# y_pred = reg.predict(X_test)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

log_model = LogisticRegression(max_iter = 3000)
log_model.fit(X_train, y_train)

y_pred = log_model.predict(X_test)

#y_pred = reg.predict(X_test)

print(f"Logistic Regression model accuracy: {metrics.accuracy_score(y_test, y_pred) * 100:.2f}%")

def sigmoid(x):
    arr = np.array(x)
    return 1/(1+((np.e)**(-arr)))


def train_logistic_regression_L2(X, y, lr=0.1, steps=1000, C=1.0):
    n_samples, n_features = X.shape
    lam = 1/(2*n_samples)
    w = np.zeros(n_features)
    b = 0.0
    for i in range(steps):

        z = X @ w + b

        y_hat = sigmoid(z)


        error = y_hat - y

        dw = X.T @ error / n_samples + 2*lam*w

        db = np.mean(error)

        w -= lr * dw
        b -= lr * db


    return w, b
