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
from my_functions import ConvertNegativeToString
from my_functions import Encoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.externals import joblib
from lightgbm import LGBMRegressor

ventas = pd.read_csv("AgentesVentas8.csv")
ventas.dropna(inplace=True)
# ventas["BKAgente"] = ventas["BKAgente"].apply(str)
ventas["BKAgente"] = ventas["BKAgente"].replace(-1, 0)
# ventas = ventas.loc[ventas["BKAgente"]>0,:]
# ventas = ventas.fillna(0)
# ventas = ventas.loc[ventas["MediaVisitasMismoDiaU10"] > 0,:]
ventas["ventas_cat"] = pd.cut(ventas["MediaVentasMismoDiaU6"],bins= 10, labels=[1,2,3,4,5,6,7,8,9,10])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(ventas,ventas["ventas_cat"]):
    strat_train_set = ventas.iloc[train_index] # !!! IMPORTANTE USAR iloc. En caso contrario aparecen NaN
    strat_test_set = ventas.iloc[test_index]

# strat_test_set.loc[strat_test_set["Ventas"].isna(),:]

for set_ in (strat_train_set,strat_test_set):
    set_.drop("ventas_cat",axis=1, inplace=True)

ventas = strat_train_set.copy()
ventas = ventas.drop("Ventas",axis=1)
ventas_labels = strat_train_set["Ventas"].copy()

# strat_train_set.loc[strat_train_set["Ventas"].isna(),:]
print("Ventas shape:",ventas.shape)
print(ventas.info())
num_pipeline = Pipeline([
    ('attribs_adder', DropColumns()),    
    ('imputer', SimpleImputer(strategy="median")),
    # ('categ',MakeCategorical()),
    ('std_scaler',RobustScaler(quantile_range=(10.,90.))),
])
# attribs = ["BKAgente","Mes","Dia","DiaSemana","DiaAnio"]
attribs = ["Mes","Dia","DiaSemana","DiaAnio"]
# print(ventas.loc[ventas["BKAgente"] < 0,:])
num_columns = ventas.drop(attribs,axis=1).columns
full_pipeline = ColumnTransformer([     
    ("num",num_pipeline,num_columns),    
    # ("cat",Encoder(one_hot_encoder=OneHotEncoder()),attribs),    
    ("cat",OneHotEncoder(categories='auto',handle_unknown='ignore'),attribs),    
])
ventas_prepared = full_pipeline.fit_transform(ventas)
print("ventas_prepared:", ventas_prepared.shape)
ventas_labels.isna()

# Training
# regressor = DecisionTreeRegressor(random_state= 42)
regressor = LGBMRegressor(random_state=42)
# final_model.fit(ventas_prepared, ventas_labels)
# param_grid = [
# {'criterion': ["mae"]}
# ]
param_grid = [
{'n_estimators': [270], 'learning_rate':[0.08,0.1]}
]
grid_search = GridSearchCV(regressor, param_grid,cv=5, scoring='neg_mean_absolute_error', return_train_score=True, verbose=10, n_jobs=8)
grid_search.fit(ventas_prepared,ventas_labels)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
print("Best:", grid_search.best_params_)
final_model = grid_search.best_estimator_

# 1st test
ventas_predictions = final_model.predict(ventas_prepared)
tree_mse = mean_squared_error(ventas_labels, ventas_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Error: ",tree_rmse)



X_test = strat_test_set.drop("Ventas",axis=1)
y_test = strat_test_set["Ventas"].copy()

X_test_prepared = full_pipeline.transform(X_test)
print(X_test_prepared.shape)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_mae = mean_absolute_error(y_test, final_predictions)

final_rmse = np.sqrt(final_mse)
print("final_rmse: ",final_rmse)
print("final_mae", final_mae)

# Save the model
joblib.dump(full_pipeline,"agentes_pipeline.joblib")
joblib.dump(final_model, "agentes_model.pkl")
