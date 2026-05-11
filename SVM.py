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
from sklearn.model_selection import train_test_split
x=d.drop(d[["target_n","flowers_n"]],axis="columns")
y=d.target_n
X_tr,X_t,y_tr,y_t=train_test_split(x,y,test_size=0.2)
from sklearn.svm import SVC
m=SVC()
m.fit(X_tr,y_tr)
print("Accuracy score: \n",m.score(X_t,y_t))
m_c=SVC(C=8)
m_c.fit(X_tr,y_tr)
print("Regularization Accuracy score: \n",m_c.score(X_t,y_t))
m_g=SVC(gamma=90)
m_g.fit(X_tr,y_tr)
print("Gamma Accuracy score: \n",m_g.score(X_t,y_t))
m_k=SVC(kernel="rbf")
m_k.fit(X_tr,y_tr)
print("Kernel Accuracy score: \n",m_k.score(X_t,y_t))