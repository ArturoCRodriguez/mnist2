#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
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

# %%
