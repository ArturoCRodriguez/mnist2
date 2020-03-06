#%%
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
#%%
ventas = pd.read_csv("AgentesVentas8.csv")
ventas.dropna(inplace=True)
ventas = ventas.loc[ventas["MediaVisitasMismoDiaU10"] > 0,:]
ventas["ventas_cat"] = pd.cut(ventas["MediaVentasMismoDiaU10"],bins= 10, labels=[1,2,3,4,5,6,7,8,9,10])

# print(ventas["ventas_cat"])
# ventas["ventas_cat"].hist()
# plt.show()

#%%

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(ventas,ventas["ventas_cat"]):
    strat_train_set = ventas.iloc[train_index]
    strat_test_set = ventas.iloc[test_index]
strat_test_set.loc[strat_test_set["Ventas"].isna(),:]
#%%
for set_ in (strat_train_set,strat_test_set):
    set_.drop("ventas_cat",axis=1, inplace=True)

ventas = strat_train_set.copy()
ventas = ventas.drop("Ventas",axis=1)
ventas_labels = strat_train_set["Ventas"].copy()
#%%
strat_train_set.loc[strat_train_set["Ventas"].isna(),:]
# print(ventas.shape)
# print(ventas_labels.shape)
#%%
ventas_labels.isna().any()

#%%
num_pipeline = Pipeline([
    ('attribs_adder', DropColumns()),
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler',StandardScaler()),
])
ventas_prepared = num_pipeline.fit_transform(ventas)
#%%
# np.isnan(ventas_prepared).any()
# type(ventas_labels)
ventas_labels.isna()
#%%
final_model = RandomForestRegressor(n_estimators=2000, max_features=48, verbose=10, n_jobs=4)
final_model.fit(ventas_prepared, ventas_labels)
ventas_predictions = final_model.predict(ventas_prepared)
tree_mse = mean_squared_error(ventas_labels, ventas_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Error: ",tree_rmse)

# %%
# param_grid = [
# {'n_estimators': [3, 10, 30,100, 300], 'max_features': [2, 4, 6, 8, 10, 12]},
# {'bootstrap': [False], 'n_estimators': [3, 10, 30], 'max_features': [2, 3, 4, 6]},
# ]
# grid_search = GridSearchCV(forest_reg, param_grid,cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=10)
# grid_search.fit(ventas_prepared,ventas_labels)
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)
# print("Best:", grid_search.best_params_)
# final_model = grid_search.best_estimator_

#%%
X_test = strat_test_set.drop("Ventas",axis=1)
y_test = strat_test_set["Ventas"].copy()
#%%
X_test_prepared = num_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

# %%
from sklearn.metrics import mean_absolute_error
final_mae = mean_absolute_error(y_test, final_predictions)
final_mae
# %%
