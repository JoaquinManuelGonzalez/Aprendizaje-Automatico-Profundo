import pandas as pd
import numpy as np
from sklearn import preprocessing

from ClassNeuronaLineal import NeuronaLineal

FILE = "../csv/automobile-simple.csv"

import chardet

with open(FILE, "rb") as f:
    result = chardet.detect(f.read())  # or readline if the file is large


df_original = pd.read_csv(FILE, encoding=result["encoding"])
#norm = preprocessing.StandardScaler()
norm = preprocessing.MinMaxScaler()

ALFA=0.01
N_ITER=30
COTAE=10e-06

columns=df_original.columns.values
features=columns[columns!="price"]
label = "price"
ppn = NeuronaLineal(n_iter=N_ITER, alpha=ALFA, cotaE=COTAE)

# map ordinal values to numerical
df_original["num-of-doors"]=df_original["num-of-doors"].map({"two":2,"four":4})

# eliminate null values by mean
null=df_original.isnull().sum()
for i in range(len(null)):
    if null[i]>0:
        ft=df_original.columns.values[i]
        df_original[ft]=df_original[ft].fillna(df_original[ft].mean())

print(df_original.head())
print(df_original.isnull().sum())

# get numerical features
numerical_features=df_original[features].select_dtypes(include=[np.number]).columns.values


df_norm=norm.fit_transform(df_original[numerical_features].values)
ITER_INDEP=50
W=np.zeros(len(numerical_features))
B=0

for i in range(ITER_INDEP):
    ppn.fit(df_norm, df_original[label].values)
    W+=ppn.w_
    B+=ppn.b_

W/=ITER_INDEP
B/=ITER_INDEP
for feature,w in zip(numerical_features,W):
   print(feature,"=",w)
