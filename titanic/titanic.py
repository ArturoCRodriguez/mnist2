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
# print(train_data[train_data["Cabin"].notnull()])
print(train_data.info())
train_data.loc[:,["Ticket","Fare","Cabin","Pclass", "Name","Survived"]].sort_values(['Fare'], ascending=False)


#%%

num_columns = ["Age","SibSp","Parch"]
cat_columns = ["Pclass","Sex","Ticket"]
cat_columns2 = ["Embarked"]
cat_columns3 = ["Fare","Cabin"]
cat_columns4 = ["Cabin"]
def get_num_of_cabins(data):
    # print(type(data["Cabin"]))    
    if(isinstance(data["Cabin"],float)):
        return 0
    else:
        return len(data["Cabin"].split())         
def get_real_fare(data):
    # print(type(data["Cabin"]))    
    if(data["num_of_cabins"] == 0):
        return 0
    else:
        return data["Fare"] / data["num_of_cabins"]
def get_deck(data):    
    if(isinstance(data["Cabin"],float)):
        return 'T'
    else:
        cabin = data["Cabin"]
        if 'A' in cabin:
            return 'A'
        elif 'B' in cabin:
            return 'B'
        elif 'C' in cabin:
            return 'C'
        elif 'D' in cabin:
            return 'D'
        elif 'E' in cabin:
            return 'E'
        elif 'F' in cabin:
            return 'F'
        elif 'G' in cabin:
            return 'G'
        return 'T'

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
class AddFeaturesFromCat(BaseEstimator,TransformerMixin):    
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):        
        X["num_of_cabins"] = X.apply(get_num_of_cabins, axis= 1)
        X["real_fare"] = X.apply(get_real_fare, axis= 1)
        # X["deck"] = X.apply(get_deck, axis= 1)
        X = X.drop(["Cabin","Fare"], axis= 1)
        return X
class GetDeck(BaseEstimator,TransformerMixin):    
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):                
        X["deck"] = X.apply(get_deck, axis= 1)
        X = X.drop(["Cabin"], axis= 1)
        return X

num_pipeline = Pipeline(
    [
        # ("columns",DropColumns(cat_columns)),
        ('imputer',  SimpleImputer(strategy="median")),
    ]
)
cat_pipeline = Pipeline(
    [        
        ('imputer_cat', SimpleImputer( strategy="most_frequent")),        
        ("cat3",OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
cat_pipeline2 = Pipeline(
    [        
        ('cabins', AddFeaturesFromCat()),        
    ]
)
cat_pipeline3 = Pipeline(
    [        
        ('decks', GetDeck()),        
        ("decks2",OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
full_pipeline = ColumnTransformer([
    ("num",num_pipeline,num_columns),
    ("add",cat_pipeline2,cat_columns3),    
    ("decks",cat_pipeline3,cat_columns4),    
    ("cat1",cat_pipeline,cat_columns2),    
    ("cat2",OneHotEncoder(handle_unknown="ignore", sparse= False),cat_columns),
    # ("cat3",OneHotEncoder(handle_unknown="ignore", sparse= False),cat_columns2),
]) 
train_data.drop(["PassengerId","Name"], axis= 1, inplace= True)
#%%
split = StratifiedShuffleSplit(n_splits=1,test_size=0.02,random_state=42)
for train_index, test_index in split.split(train_data,train_data["Sex"]):
    train_data_train = train_data.iloc[train_index]
    val_data = train_data.iloc[test_index]
#%%

X_train = train_data_train.drop("Survived", axis=1)
y_train = train_data_train["Survived"]
X_val = val_data.drop("Survived", axis=1)
y_val = val_data["Survived"]

X_train_prepared = full_pipeline.fit_transform(X_train)
X_val_prepared = full_pipeline.transform(X_val)

#%%
clf = RandomForestClassifier(random_state=42,n_estimators=300, n_jobs=8)
clf2 = LGBMClassifier()
clf3 = MLPClassifier(random_state=43, max_iter=600)

# plot_learning_curves(clf,full_pipeline,X_train,X_val,y_train,y_val)
# plt.show()
#%%
clf.fit(X_train_prepared, y_train)
clf2.fit(X_train_prepared,y_train)
clf3.fit(X_train_prepared, y_train)

def print_results(model_name,y_true, y_predict):
    print("{} results:".format(model_name))
    auc = roc_auc_score(y_true,y_predict)
    f1 = f1_score(y_true,y_predict)
    precision = precision_score(y_true,y_predict)
    recall = recall_score(y_true,y_predict)
    accuracy = accuracy_score(y_true,y_predict)
    print("auc\t f1\t precision\t recall\t accuracy")
    print("{}\t {}\t {}\t {}\t {}".format(auc,f1,precision,recall,accuracy))

y_pred = clf.predict(X_val_prepared)
y_pred2 = clf2.predict(X_val_prepared)
y_pred3 = clf3.predict(X_val_prepared)

print_results("Forest",y_val, y_pred)
print_results("LightGBM",y_val, y_pred2)
print_results("NN",y_val, y_pred3)
