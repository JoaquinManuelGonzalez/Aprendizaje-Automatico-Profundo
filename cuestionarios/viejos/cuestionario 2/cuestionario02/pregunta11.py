import numpy as np
import pandas as pd
from sklearn import preprocessing
W_diametro=2.3944
W_color=-2.3891
b=-0.06368

FILE="../csv/FrutasTrain.csv"

df=pd.read_csv(FILE)
norm=preprocessing.StandardScaler()

features=["Diametro","Color"]
df[features]=norm.fit_transform(df[features])

# inciso a
indefinido=0
for diam,color,clase in zip(df["Diametro"],df["Color"],df["Clase"]):
    linea=W_diametro*diam+W_color*color+b
    linea=(2.0/(1+np.exp(-2*linea))-1)
    #predict=2*(linea>0)*1-1
    print(f"Diametro: {diam:3f} Color: {color:3f} Clase: {clase} Res: {linea:3f}")
    if linea<0.8 and linea>-0.8:
        indefinido+=1
print(f"Indefinidos: {indefinido}")

# inciso b
linea=W_diametro*12+W_color*150+b
linea=(2.0/(1+np.exp(-2*linea))-1)

print(f"Inciso b: {linea:3f}")

# inciso c
linea=W_diametro*17+W_color*90+b
linea=(2.0/(1+np.exp(-2*linea))-1)

print(f"Inciso c: {linea:3f}")
