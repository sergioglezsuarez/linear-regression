from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("tiempos.csv", sep=";")


def conversion(column):
    lista = []
    for tiempo in column:
        h = int(tiempo[:2])
        m = int(tiempo[3:5])
        lista.append(h * 60 + m)
    return pd.Series(lista)


data = data.apply(conversion)
train, test = train_test_split(data, train_size=0.7)

linear_centered = LinearRegression().fit(train.iloc[:, :6], train["T_Final"])
y_true = train["T_Final"]
y_pred = linear_centered.predict(train.iloc[:, :6])

linear = LinearRegression(fit_intercept=False).fit(train.iloc[:, :6], train["T_Final"])
y_true = train["T_Final"]
y_pred = linear.predict(train.iloc[:, :6])

labels = ["w0", "w1", "w2", "w3", "w4", "w5", "w6"]

data = [np.concatenate([[linear.intercept_], linear.coef_]), np.concatenate([[linear_centered.intercept_], linear_centered.coef_])]

X = np.arange(7)
fig, ax = plt.subplots()
ax.bar(X, data[0], color='b', width=0.25, label='Regresión lineal no centrada')
ax.bar(X + 0.25, data[1], color='r', width=0.25, label='Regresión lineal centrada')
ax.set_xticks(X, labels=labels)
ax.legend()
plt.show()

errors = []
errors_centered = []

alphas = [0.000001, 1, 10, 100, 1000, 10000, 50000]

for i in alphas:
    l2 = Ridge(alpha=i, fit_intercept=False).fit(train.iloc[:, :6], train["T_Final"])
    l2_centered = Ridge(alpha=i).fit(train.iloc[:, :6], train["T_Final"])
    errors.append(mean_squared_error(y_true, l2.predict(train.iloc[:, :6]), squared=False))
    errors_centered.append(mean_squared_error(y_true, l2_centered.predict(train.iloc[:, :6]), squared=False))

print(errors)
print(errors_centered)

print(pd.DataFrame([errors], columns=alphas))
print(pd.DataFrame([errors_centered], columns=alphas))

l21 = Ridge(alpha=10, fit_intercept=False).fit(train.iloc[:, :6], train["T_Final"])
l22 = Ridge(alpha=10000, fit_intercept=False).fit(train.iloc[:, :6], train["T_Final"])
l21_centered = Ridge(alpha=10).fit(train.iloc[:, :6], train["T_Final"])
l22_centered = Ridge(alpha=10000).fit(train.iloc[:, :6], train["T_Final"])

data = [np.concatenate([[l21.intercept_], l21.coef_]), np.concatenate([[l21_centered.intercept_], l21_centered.coef_]),
        np.concatenate([[l22.intercept_], l22.coef_]), np.concatenate([[l22_centered.intercept_], l22_centered.coef_])]

X = np.arange(7)
fig, ax = plt.subplots()
ax.bar(X - 0.3, data[0], color='b', width=0.2, label='L2 no centrada con alpha=10')
ax.bar(X - 0.1, data[1], color='r', width=0.2, label='L2 centrada con alpha=10')
ax.bar(X + 0.1, data[2], color='g', width=0.2, label='L2 no centrada con alpha=10000')
ax.bar(X + 0.3, data[3], color='y', width=0.2, label='L2 centrada con alpha=10000')
ax.set_xticks(X, labels=labels)
ax.legend()
plt.show()

errors = []
errors_centered = []

alphas = [0.01, 1, 10, 100, 1000, 5000, 10000]

for i in alphas:
    l2 = Lasso(alpha=i, fit_intercept=False, max_iter=10000).fit(train.iloc[:, :6], train["T_Final"])
    l2_centered = Lasso(alpha=i, max_iter=10000).fit(train.iloc[:, :6], train["T_Final"])
    errors.append(mean_squared_error(y_true, l2.predict(train.iloc[:, :6]), squared=False))
    errors_centered.append(mean_squared_error(y_true, l2_centered.predict(train.iloc[:, :6]), squared=False))

print(errors)
print(errors_centered)

print(pd.DataFrame([errors], columns=alphas))
print(pd.DataFrame([errors_centered], columns=alphas))

l11 = Lasso(alpha=1, fit_intercept=False, max_iter=10000).fit(train.iloc[:, :6], train["T_Final"])
l12 = Lasso(alpha=10, fit_intercept=False, max_iter=10000).fit(train.iloc[:, :6], train["T_Final"])
l11_centered = Lasso(alpha=1, max_iter=10000).fit(train.iloc[:, :6], train["T_Final"])
l12_centered = Lasso(alpha=10, max_iter=10000).fit(train.iloc[:, :6], train["T_Final"])

data = [np.concatenate([[l11.intercept_], l11.coef_]), np.concatenate([[l11_centered.intercept_], l11_centered.coef_]),
        np.concatenate([[l12.intercept_], l12.coef_]), np.concatenate([[l12_centered.intercept_], l12_centered.coef_])]

X = np.arange(7)
fig, ax = plt.subplots()
ax.bar(X - 0.3, data[0], color='b', width=0.2, label='L1 no centrada con alpha=10')
ax.bar(X - 0.1, data[1], color='r', width=0.2, label='L1 centrada con alpha=10')
ax.bar(X + 0.1, data[2], color='g', width=0.2, label='L1 no centrada con alpha=50')
ax.bar(X + 0.3, data[3], color='y', width=0.2, label='L1 centrada con alpha=50')
ax.set_xticks(X, labels=labels)
ax.legend()
plt.show()