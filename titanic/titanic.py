#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
here = os.path.dirname(os.path.abspath(__file__))

train_data = pd.read_csv( os.path.join(here,"train.csv"))
test_data = pd.read_csv( os.path.join(here,"test.csv"))
gender_data = pd.read_csv( os.path.join(here,"gender_submission.csv"))
print(train_data[train_data["Embarked"].isnull()])
train_data = train_data.fillna(np.nan)


#%%

num_columns = ["Age","SibSp","Parch","Fare"]
cat_columns = ["Pclass","Sex","Ticket"]
cat_columns2 = ["Embarked"]
class DropColumns(BaseEstimator,TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):
        X = X.drop(self.columns, axis = 1)
        return X
class Prueba(BaseEstimator,TransformerMixin):
    def __init__(self, columns = []):
        self.columns = columns
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):
        print(X)
        return X
num_pipeline = Pipeline(
    [
        # ("columns",DropColumns(cat_columns)),
        ('imputer',  SimpleImputer(strategy="mean")),
    ]
)
cat_pipeline = Pipeline(
    [
        # ("columns",DropColumns(num_columns)),
        ('imputer_cat', SimpleImputer( strategy="most_frequent")),
        ('prueba', Prueba()),
        ("cat3",OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_columns),
    ("cat1",cat_pipeline,cat_columns2),    
    ("cat2",OneHotEncoder(handle_unknown="ignore", sparse= False),cat_columns),
    # ("cat3",OneHotEncoder(handle_unknown="ignore", sparse= False),cat_columns2),
]) 
train_data.drop(["PassengerId","Name","Cabin"], axis= 1, inplace= True)

#%%

X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]

X_train_prepared = full_pipeline.fit_transform(X_train)


#%%
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_prepared, y_train)
y_pred = clf.predict(X_train_prepared)
print(roc_auc_score(y_train,y_pred))
