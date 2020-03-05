import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from my_functions import DropColumns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

ventas = pd.read_csv("AgentesVentas8.csv")
ventas.dropna(inplace=True)
ventas = ventas.loc[ventas["MediaVisitasMismoDiaU10"] > 0,:]
ventas["ventas_cat"] = pd.cut(ventas["MediaVentasMismoDiaU10"],bins= 10, labels=[1,2,3,4,5,6,7,8,9,10])
# print(ventas["ventas_cat"])
# ventas["ventas_cat"].hist()
# plt.show()


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(ventas,ventas["ventas_cat"]):
    strat_train_set = ventas.loc[train_index]
    strat_test_set = ventas.loc[test_index]
for set_ in (strat_train_set,strat_test_set):
    set_.drop("ventas_cat",axis=1, inplace=True)

ventas = strat_train_set.copy()
ventas = ventas.drop("Ventas",axis=1)
ventas_labels = strat_train_set["Ventas"].copy()

# print(ventas.shape)
# print(ventas_labels.shape)

num_pipeline = Pipeline([
    ('attribs_adder', DropColumns()),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])
ventas_prepared = num_pipeline.fit_transform(ventas)

forest_reg = RandomForestRegressor()
forest_reg.fit(ventas_prepared, ventas_labels)
ventas_predictions = forest_reg.predict(ventas_prepared)
tree_mse = mean_squared_error(ventas_labels, ventas_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Error: ",tree_rmse)