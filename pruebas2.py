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
from my_functions import MakeCategorical
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
ventas = pd.read_csv("AgentesVentas8.csv")

# ventas = ventas.loc[:100,["DiaSemana"]]
ventas = pd.DataFrame.fillna(ventas,0)
num_pipeline = Pipeline([
    ('attribs_adder', DropColumns()),
    ('imputer', SimpleImputer(strategy="median")),
    # ('categ',MakeCategorical()),
    ('std_scaler',RobustScaler()),
])
attribs = ["BKAgente","Mes","Dia","DiaSemana","DiaAnio"]

full_pipeline = ColumnTransformer([    
    ("num",num_pipeline,ventas.columns),    
    ("cat",OneHotEncoder(),attribs),    
])

ventas = full_pipeline.fit_transform(ventas)
regressor = MLPRegressor(random_state=42, max_iter=)

print((ventas.toarray()))
