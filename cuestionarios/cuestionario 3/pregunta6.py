from sklearn.neural_network import MLPClassifier
import pandas as pd
import chardet
from sklearn import preprocessing

FILE = "../csv/Sonar.csv"

with open(FILE, "rb") as f:
    result = chardet.detect(f.read())

df = pd.read_csv(FILE, encoding=result["encoding"])
label = "class"
features = df.columns.drop(label)

norm = preprocessing.StandardScaler()

X = df[features].values
y = df[label].values

X = norm.fit_transform(X)

clf = MLPClassifier(
    solver="sgd",
    learning_rate_init=0.1,
    hidden_layer_sizes=(10,),
    max_iter=2000,
    verbose=False,
    tol=10e-05,
    activation="tanh",
)

model = clf.fit(X, y)
print("Score: ", model.score(X, y))
