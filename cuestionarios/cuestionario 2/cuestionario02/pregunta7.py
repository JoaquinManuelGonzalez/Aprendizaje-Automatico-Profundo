from ClassNeuronaLineal import NeuronaLineal
import numpy as np
import pandas as pd
from matplotlib import pylab as plt
from grafica import *
from sklearn import preprocessing

# Datos de entrada
# one half 0, one half 1

X1 = [0, 0, 0, 0, 1, 1, 1, 1]
X2 = [0, 0, 1, 1, 0, 0, 1, 1]
X3 = [0, 1, 0, 1, 0, 1, 0, 1]
Y = [0, 1, 2, 3, 4, 5, 6, 7]
df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3, "Y": Y})


# Crear objeto neurona
# inciso a
ALFA=0.1
MAX_ITE=500
COTA_ERROR=10e-20
ppn = NeuronaLineal(alpha=ALFA,n_iter=MAX_ITE,cotaE=COTA_ERROR)

# Entrenar neurona
features = ["X1", "X2", "X3"]
ppn.fit(df[features].values, df["Y"].values.reshape(-1, 1))
print("Pesos finales: %s" % ppn.w_, ppn.b_)
print("Iteraciones: %d" % len(ppn.errors_))

plt.show()
