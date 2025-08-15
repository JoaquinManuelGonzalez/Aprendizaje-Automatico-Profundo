import pandas as pd
from ClassPerceptron import Perceptron

# get encoding of csv file
import chardet
with open("../csv/Lentes.csv", 'rb') as f:
    result = chardet.detect(f.read())  # or readline if the file is large


df=pd.read_csv("../csv/Lentes.csv",encoding=result['encoding'])

features=df.columns[1:-1].tolist()
labels=df.columns[-1]
print("features:",features)
print("labels:",labels)

df["Edad"]=df["Edad"].map({"Joven":0,"pre_presb":1,"Presbicia":2})
df["Prescripcion"]=df["Prescripcion"].map({"Hipermetropía":0,"Miopía":1})
df["Astigmatismo"]=df["Astigmatismo"].map({"NO":0,"SI":1})
df["Lagrimas"]=df["Lagrimas"].map({"Normal":0,"Reducida":1})
df["Diagnostico"]=df["Diagnostico"].map({"Lentes_Blandos":1,"Lentes_Duros":0,"No_usar_Lentes":0})

ppn=Perceptron(alpha=0.05,n_iter=100,title=features)
ppn.fit(df[features].values,df[labels].values)

df_to_test=df[df[labels]==1]
res=ppn.predict(df_to_test[features].values)
print("res:",res)


