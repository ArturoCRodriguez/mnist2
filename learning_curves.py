#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size= 0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m],y_train_predict))
        val_errors.append(mean_squared_error(y_val,y_val_predict))
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="val")

m = 1000
X = 6 * np.random.rand(m,1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m,1)

m_test = 50
X_test = 6 * np.random.rand(m,1) - 3
y_test = 0.5 * X_test**2 + X_test + 2 + np.random.randn(m,1)
#%%
poly_features = PolynomialFeatures(degree=20, include_bias=False)
print(X.shape)
X_poly = poly_features.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
a = np.linspace(-3, 3, 100).reshape(100,1)
print("a:",a.shape)
a_poly = poly_features.transform(a)
b = lin_reg.predict(a_poly)

plt.plot(X,y,'bo')
plt.plot(a,b,'r')
plt.show()

#%%
model = LinearRegression()
plot_learning_curves(model,X,y)

# %%
