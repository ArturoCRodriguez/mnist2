import pandas as pd
import matplotlib.pyplot as plt
housing = pd.read_csv("housing.csv")
print(housing.info())
print(housing.describe())
housing.hist(bins = 50, figsize = (20,15))
plt.show()