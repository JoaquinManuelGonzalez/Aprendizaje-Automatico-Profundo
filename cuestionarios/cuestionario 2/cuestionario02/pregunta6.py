import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ClassNeuronaLineal import NeuronaLineal

m=np.array([2,3,4,4,5,6,6,7,7,8,10,10])
f=np.array([1,3,2,4,4,4,6,4,6,7,9,10])

df=pd.DataFrame({'m':m,'f':f})

print(df.corr())

# sparse plot
df.plot.scatter(x='m',y='f')

ppn=NeuronaLineal(alpha=0.01, n_iter=100,cotaE=0.0001,draw=0,title=['m','f'])

ppn.fit(df['m'].values.reshape(-1,1),df['f'].values)
print(ppn.w_,ppn.b_)
#plt.show()

feature='m'
label='f'
w = [1.0196201,0.91090581,0.79816056]
b = [-0.18606774,-0.30860262,0.52217544]
errors=[]
for wi,bi in zip(w,b):
    sum=0
    for atj,pej in zip(df[feature].values,df[label].values):
        sum+=(pej-(wi*atj+bi))**2
    errors.append(sum/len(df[feature].values))

for wi,bi,error in zip(w,b,errors):
    print("\(w=",wi,"\quadsep","b=","\quadsep",bi,"\\text{ECM}:%.2f\)"%error)
