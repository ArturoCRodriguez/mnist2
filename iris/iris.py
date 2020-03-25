from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
iris = datasets.load_iris()
# print(list(iris.keys()))
X = iris["data"][:,3:]
y = (iris["target"] == 2).astype(np.int)

log_reg = LogisticRegression()
log_reg.fit(X,y)
X_new = np.linspace(0,3,1000).reshape(-1,1)
y_proba = log_reg.predict_proba(X_new)
print(y_proba) # En y_proba se devuelven 2 columnas "0" y "1" y en cada una tenemos las probabilidades de ser 0 o 1 para cada row
# prueba = y_proba[:,0] + y_proba[:,1]
# print(prueba) # la suma de ambas columnas es 1
plt.plot(X_new, y_proba[:,1],"g-",label="Iris-Virginica")
plt.plot(X_new, y_proba[:,0],"b--", label = "Not Iris-Virginica")
plt.show()