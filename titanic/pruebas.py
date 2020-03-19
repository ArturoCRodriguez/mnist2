#%%
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from titanic_functions import plot_learning_curves

here = os.path.dirname(os.path.abspath(__file__))

train_data = pd.read_csv( os.path.join(here,"train.csv"))
test_data = pd.read_csv( os.path.join(here,"test.csv"))
gender_data = pd.read_csv( os.path.join(here,"gender_submission.csv"))
# print(train_data[train_data["Embarked"].isnull()])
pd.set_option('display.max_rows', None)

data = train_data.loc[:,["Pclass","Sex"]]
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
data_prepared = encoder.fit_transform(data)
data_prepared.shape