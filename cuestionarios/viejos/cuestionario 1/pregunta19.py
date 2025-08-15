import pandas as pd
from ClassPerceptron import Perceptron

# get encoding of csv file
FILE="../csv/Mushroom.csv"

import chardet
with open(FILE, 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


df=pd.read_csv(FILE,encoding=result['encoding'])

print(df.head())
df.rename(columns={" bruises?":"bruises"},inplace=True)
df.rename(columns={" odor":"odor"},inplace=True)
df.rename(columns={" gill-size":"gill-size"},inplace=True)
df.rename(columns={" spore-print-color":"spore-print-color"},inplace=True)
df.rename(columns={" stalk-surface-below-ring":"stalk-surface-below-ring"},inplace=True)

features=["odor","gill-size","bruises","spore-print-color","stalk-surface-below-ring"]
labels="Tipo"

df["Tipo"]=df["Tipo"].map({"p":1,"e":0})
for feature in features:
    uniques=df[feature].unique()
    for i in range(len(uniques)):
        df[feature]=df[feature].replace(uniques[i],i)

ppn=Perceptron(alpha=0.05,n_iter=100,title=features)
ppn.fit(df[features].values,df[labels].values)

df_to_test=df[df[labels]==1]
res=ppn.predict(df_to_test[features].values)
print("res:",res.sum()/len(res))


