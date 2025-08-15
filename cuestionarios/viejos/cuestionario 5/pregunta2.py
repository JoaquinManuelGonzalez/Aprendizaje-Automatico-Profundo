import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("~/facultad/4to_anio/DEEPL/csv/clima_numerico.csv")
df["Juega"].replace({"Si": 1, "No": 0}, inplace=True)
X = df.iloc[:,:-1]
y = df["Juega"]

norm = MinMaxScaler()
X = norm.fit_transform(X)

W = [
    [1.104, -0.057, -0.345, 0.016],
    [0.026, -0.006, -0.142, -0.024],
    [0.092, 0.003, -0.412, 0.029]
]

b = [0.154, 0.114, 0.391]

entry = [[1.0, 2.0, 25.0, 1.0]]
entry = norm.transform(entry)

for i in range(len(W)):

    print(f"Perceptron {i + 1}:")

    y_hat = np.where((np.dot(X, W[i]) + b[i]) >= 0.0, 1, 0)
    print(y_hat)

    print(f"Resultado para la nueva entrada: {int((np.dot(entry[0], W[i]) + b[i]) >= 0.0)}")




