import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
l=load_iris()
print(l)
print(l.feature_names)
print(l.data)
print(l.target_names)
d=pd.DataFrame(l.data,columns=l.feature_names)
print(d.head())
d["target_n"]=l.target
print(d)
d["flowers_n"]=d.target_n.apply(lambda x: l.target_names[x])
print(d)
d0=d[d.target_n==0]
d1=d[d.target_n==1]
d2=d[d.target_n==2]
print(d0)
print(d1)
print(d2)
plt.scatter(d0["sepal length (cm)"],d0["sepal width (cm)"],color="b",marker="+")
plt.scatter(d1["petal length (cm)"],d1["petal width (cm)"],color="r",marker=".")
plt.show()
plt.scatter(d2["petal length (cm)"],d2["petal width (cm)"],color="g",marker=".")
plt.scatter(d1["sepal length (cm)"],d1["sepal width (cm)"],color="orange",marker="+")
plt.show()