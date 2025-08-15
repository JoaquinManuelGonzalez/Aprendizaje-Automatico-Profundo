import pandas as pd
import numpy as np
from sklearn import preprocessing

from ClassNeuronaLineal import NeuronaLineal

FILE = "../csv/automobile-simple.csv"

import chardet

with open(FILE, "rb") as f:
    result = chardet.detect(f.read())  # or readline if the file is large


df_original = pd.read_csv(FILE, encoding=result["encoding"])
norm = preprocessing.MinMaxScaler()
#norm = preprocessing.StandardScaler()


columns=df_original.columns.values
features=columns[columns!="price"]
label = "price"
ALFA=0.01
N_ITER=30
COTAE=10e-06
ppn = NeuronaLineal(n_iter=N_ITER,alpha=ALFA,cotaE=COTAE)

# eliminate null values by mean
df_original[label]=df_original[label].fillna(df_original[label].mean())
df_original["num-of-doors"]=df_original["num-of-doors"].map({"two":2,"four":4})
df_original["num-of-doors"]=df_original["num-of-doors"].fillna(df_original["num-of-doors"].mean())
df_original["horsepower"]=df_original["horsepower"].fillna(df_original["horsepower"].mean())

# get numerical features
numerical_features=df_original[features].select_dtypes(include=[np.number]).columns.values


df_norm=norm.fit_transform(df_original[numerical_features].values)
#df_norm=df_original[numerical_features].values # sin normalizar

ppn.fit(df_norm, df_original[label].values)
for feature,w in zip(numerical_features,ppn.w_):
    print(feature,"=",w)
