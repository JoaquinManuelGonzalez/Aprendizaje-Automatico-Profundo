import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ClassNeuronaLineal import NeuronaLineal
features=["AT","V","AP","RH"]
label=["PE"]
columns=features+label

df=pd.read_csv('../csv/CCPP.csv', sep=',')

norm=MinMaxScaler().fit(df[columns])
X=norm.fit_transform(df[features])
Y=df[label].values
ITERACIONES=5
ppn=NeuronaLineal(alpha=0.01, n_iter=1000)
W=[]
for i in range(ITERACIONES):
    print("Iteracion: ",i)
    ppn.fit(X,Y)
    W.append(ppn.w_)

# average of all weights
W=np.array(W)
W=W.mean(axis=0)
print(W)

print(df.head())
