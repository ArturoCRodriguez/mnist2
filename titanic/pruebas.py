#%%
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from titanic_functions import plot_learning_curves,get_num_of_cabins,get_real_fare,get_deck,DropColumns,AddFeaturesFromCat,GetDeck, print_results

here = os.path.dirname(os.path.abspath(__file__))

d = {
    'col1':[1,10,5,2,17,3,27], 
    'col2':['a','b','np.nan','c','d','e','f'],
    'col3':['sd','asdb','asdas','cgh','fgh','fgh','werf'],
    'col4':[0.2,0.10,0.35,0.52,0.6917,30,0.627]
    }
data = pd.DataFrame(data= d)
data
#%%
pipeline1 = Pipeline([
    ("1",OneHotEncoder(handle_unknown='ignore',sparse=False))
])
pipeline2 = Pipeline([
    ("1",DropColumns(["col1"]))
])
transformer = ColumnTransformer([
    # ("p1",pipeline1,data.columns),
    ("p2",pipeline2,data.columns)    
])
full = Pipeline([
    ("f1",transformer)
])
data_prepared = full.fit_transform(data)
type(data_prepared)