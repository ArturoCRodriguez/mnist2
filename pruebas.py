#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
housing = pd.read_csv("housing.csv")
print(type(housing[["latitude"]]))
print(type(housing["latitude"]))

# %%
