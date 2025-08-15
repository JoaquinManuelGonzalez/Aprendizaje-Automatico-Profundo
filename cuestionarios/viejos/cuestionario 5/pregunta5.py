import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from keras.utils import to_categorical
from keras.layers import Dense
from keras.models import Sequential

df = pd.read_csv("../csv/Iris.csv")
features = df.columns[0:4].tolist()
label = df.columns[4]

print(df.head())
norm = StandardScaler()
X = df[features].values
X=norm.fit_transform(df[features].values)
y = df[label].values
encoder = preprocessing.LabelEncoder()
y = encoder.fit_transform(y)
y = to_categorical(y)
print(X.shape)


W = np.array(
    [
        [0.28207211, 0.82365046, -1.6953921, -0.43556031],
        [0.19121698, -0.58137596, -0.1947324, -0.07381981],
        [-0.94732232, -0.56692314, 1.19278561, 0.79934034],
    ]
)

b = np.array([[-0.03205492], [0.23094765], [-0.69461625]])

print(W.shape)
# add b column to W
asd = np.append(W, b, axis=1)
print(asd.shape)
print("ASd^")
print(asd)

model = Sequential()
model.add(Dense(3, input_dim=4, activation="softmax"))
model.layers[0].set_weights(
    [
        W.T,
        b.reshape(
            3,
        ),
    ]
)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
y_pred = model.predict(X)

print(y_pred)
# accuracy
acc = np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / y.shape[0]
print(acc)

# iii
entry = [5.92, 6.0, 12.0, 2.5]
entry=norm.transform([entry])
y_pred = model.predict(entry)
# convert y_pred to class labels
y_pred = np.argmax(y_pred, axis=1)
# reverse encoding
y_pred = encoder.inverse_transform(y_pred)
print(y_pred)
