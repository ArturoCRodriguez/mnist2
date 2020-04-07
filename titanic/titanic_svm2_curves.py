#%%
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder,RobustScaler, Normalizer, OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from titanic_functions import plot_learning_curves,get_num_of_cabins,get_real_fare,get_deck,DropColumns
from titanic_functions import AddFeaturesFromCat,GetDeck, print_results,Combine, CalculateFamilySize,AgeBins
from sklearn.preprocessing import KBinsDiscretizer
import xgboost as xgb
import re
import random
here = os.path.dirname(os.path.abspath(__file__))

train_data = pd.read_csv( os.path.join(here,"train.csv"))
test_data = pd.read_csv( os.path.join(here,"test.csv"))
test_data_survived = pd.read_csv( os.path.join(here,"test_survived.csv"))
# titles = ["Dona","Mrs","Mr","Miss","Master","Rev","Ms","Dr","Don","Major","Col","Capt"]
titles = ["Mrs","Mr","Miss","Master","Ms","Mlle","Mme"]
pattern = r".*\(.*\).*"
embarked_most_common = train_data["Embarked"].mode()
mean_fare_by_class = train_data.groupby("Pclass")["Fare"].median()

def get_title(name):
    for t in titles:
        if t in name:
            return t
    return "Rare"
def get_hascarer(name):
    if re.match(pattern,name) is not None:
        return 1
    else:
        return 0

#%%

train_data["Title"] = train_data["Name"].apply(get_title)
train_data["HasCarer"] = train_data["Name"].apply(get_hascarer)

test_data["Title"] = test_data["Name"].apply(get_title)
test_data["HasCarer"] = test_data["Name"].apply(get_hascarer)

test_data_survived["Title"] = test_data_survived["Name"].apply(get_title)
test_data_survived["HasCarer"] = test_data_survived["Name"].apply(get_hascarer)
# train_data.loc[:,["Name","HasCarer"]].sort_values("HasCarer")

train_data['Title'] = train_data['Title'].replace('Mlle','Miss')
train_data['Title'] = train_data['Title'].replace('Ms','Miss')
train_data['Title'] = train_data['Title'].replace('Mme','Mrs')

test_data['Title'] = test_data['Title'].replace('Mlle','Miss')
test_data['Title'] = test_data['Title'].replace('Ms','Miss')
test_data['Title'] = test_data['Title'].replace('Mme','Mrs')

test_data_survived['Title'] = test_data_survived['Title'].replace('Mlle','Miss')
test_data_survived['Title'] = test_data_survived['Title'].replace('Ms','Miss')
test_data_survived['Title'] = test_data_survived['Title'].replace('Mme','Mrs')


#%%
mean_ages_by_title = train_data.groupby("Title")["Age"].mean()
std_age_by_title = train_data.groupby("Title")["Age"].std()
std_age_by_title.fillna(0,inplace=True)
#%%
# train_age_null = train_data["Age"].isnull().sum()
# test_age_null = test_data["Age"].isnull().sum()

train_data["Age"] = train_data.apply(lambda x: x["Age"] if not np.isnan(x["Age"]) else random.uniform(-1,1)*std_age_by_title[x["Title"]]+mean_ages_by_title[x["Title"]] , axis=1)
test_data["Age"] = test_data.apply(lambda x: x["Age"] if not np.isnan(x["Age"]) else random.uniform(-1,1)*std_age_by_title[x["Title"]]+mean_ages_by_title[x["Title"]] , axis=1)
test_data_survived["Age"] = test_data_survived.apply(lambda x: x["Age"] if not np.isnan(x["Age"]) else random.uniform(-1,1)*std_age_by_title[x["Title"]]+mean_ages_by_title[x["Title"]] , axis=1)
# %%

train_data["Fare"] = train_data.apply(lambda x: x["Fare"] if not np.isnan(x["Fare"]) else mean_fare_by_class[x["Pclass"]],axis= 1)
test_data["Fare"] = test_data.apply(lambda x: x["Fare"] if not np.isnan(x["Fare"]) else mean_fare_by_class[x["Pclass"]],axis= 1)
test_data_survived["Fare"] = test_data_survived.apply(lambda x: x["Fare"] if not np.isnan(x["Fare"]) else mean_fare_by_class[x["Pclass"]],axis= 1)
# %%
train_data["HasCabin"] = train_data["Cabin"].apply(lambda x: 0 if isinstance(x,float) else 1)
test_data["HasCabin"] = test_data["Cabin"].apply(lambda x: 0 if isinstance(x,float) else 1)
test_data_survived["HasCabin"] = test_data_survived["Cabin"].apply(lambda x: 0 if isinstance(x,float) else 1)
#%%
train_data["Fare_cat"] = pd.cut(train_data["Fare"],bins=[-1.,7.910400,14.454200,31.000000,np.inf],labels=[0,1,2,3])
test_data["Fare_cat"] = pd.cut(test_data["Fare"],bins=[-1.,7.910400,14.454200,31.000000,np.inf],labels=[0,1,2,3])
test_data_survived["Fare_cat"] = pd.cut(test_data_survived["Fare"],bins=[-1.,7.910400,14.454200,31.000000,np.inf],labels=[0,1,2,3])
#%%
train_data["IsAlone"] = train_data.apply(lambda x: 1 if x["SibSp"]+x["Parch"] > 0 else 0 ,axis=1)
test_data["IsAlone"] = test_data.apply(lambda x: 1 if x["SibSp"]+x["Parch"] > 0 else 0 ,axis=1)
test_data_survived["IsAlone"] = test_data_survived.apply(lambda x: 1 if x["SibSp"]+x["Parch"] > 0 else 0 ,axis=1)
#%%
train_data["FamiliSize"] = train_data.apply(lambda x: x["SibSp"]+x["Parch"] ,axis=1)
test_data["FamiliSize"] = test_data.apply(lambda x: x["SibSp"]+x["Parch"] ,axis=1)
test_data_survived["FamiliSize"] = test_data_survived.apply(lambda x: x["SibSp"]+x["Parch"] ,axis=1)
#%%
train_data["Embarked"] = train_data["Embarked"].apply(lambda x: "S" if isinstance(x,float) else x)
test_data["Embarked"] = test_data["Embarked"].apply(lambda x: "S" if isinstance(x,float) > 0 else x)
test_data_survived["Embarked"] = test_data_survived["Embarked"].apply(lambda x: "S" if isinstance(x,float) > 0 else x)
#%%
train_data["Deck"] = train_data.apply(get_deck,axis=1)
test_data["Deck"] = test_data.apply(get_deck,axis=1)
test_data_survived["Deck"] = test_data_survived.apply(get_deck,axis=1)
#%%
print(train_data[["HasCabin","Survived"]].groupby(["HasCabin"],as_index=False).mean())
print(test_data_survived[["HasCabin","Survived"]].groupby(["HasCabin"],as_index=False).mean())
#%%
X_train = train_data.drop("Survived",axis=1)
y_train = train_data["Survived"]
X_test_survived = test_data_survived.drop("Survived",axis=1)
y_test_survived = test_data_survived["Survived"]
X_test = test_data
#%%
print(X_train.info())
columns_to_drop = ["PassengerId","Name","Cabin","Fare_cat","Parch","SibSp","Deck"]
num_columns = ["Age","Fare","FamiliSize"]
cat_columns_stay = ["Pclass","Sex","HasCarer","Title","IsAlone","Embarked","HasCabin","Ticket"]

num_pipeline = Pipeline(
    [
        # ('std_scaler',StandardScaler()),           
        # ('std_scaler',RobustScaler()),           
        ('std_scaler',MinMaxScaler()),           
    ]
)
cat_stay_pipeline = Pipeline(
    [
        ("cat3",OneHotEncoder(handle_unknown="ignore",sparse= False))
    ]
)
transformer = ColumnTransformer([
    ("num",num_pipeline,num_columns),    
    ("cat_stay",cat_stay_pipeline,cat_columns_stay),    
])
full_pipeline = Pipeline([
    ("1",DropColumns(columns_to_drop)),
    ("2",transformer)
]) 
# test_data.loc[test_data["Fare_cat"].isnull(),:]
X_train_prepared = full_pipeline.fit_transform(X_train)
X_test_prepared = full_pipeline.transform(X_test)
X_test_survived_prepared = full_pipeline.transform(X_test_survived)
#%%
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf2 = LGBMClassifier(n_jobs=4)
clf3 = MLPClassifier(random_state=42)
clf4 = AdaBoostClassifier(random_state=42)
clf5 = xgb.XGBClassifier(random_state=42)
clf6 = SVC(kernel="rbf",random_state=42)
# params = [{"gamma":[0.001,0.003,0.01,0.03,0.05,0.07,0.1,0.15,0.2,0.3,1,3,10], "C":[0.1,0.3,1,3,10,30,80,90,100,110,150,200,300,500,1000]}]
params = [{"gamma":[0.001,0.003,0.01,0.03,0.07,0.1,0.15,0.2], "C":[10,30,50,70,75,80,90]}]
# params = [{
#     # 'max_depth': [10, 30,  50, 70, 90, None],
#     # 'criterion':["entropy","gini"],
#     'n_estimators':[10,20,30,40,50,70,100,250,370, 500,760, 800,830,900]}
#     ]
# params = [{'n_estimators':[500,510,530,550,570,600]}]
# params = [{'max_iter':[100,200,270,300,400]}]
# params = [
#     {'solver': ["lbfgs","adam"],'activation':["relu","logistic"], 'max_iter':[100,200,270,300,400], 'alpha': [0.01,0.03,0.1,0.3], 'learning_rate_init':[0.01,0.03,0.1,0.3]}
# ]
grid = GridSearchCV(estimator = clf6, param_grid= params, scoring='roc_auc',n_jobs=-1,cv=20, verbose=100)
grid.fit(X_train_prepared,y_train)
cvres = grid.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
print("Best:", grid.best_params_)
final_model = grid.best_estimator_
y_train_pred = final_model.predict(X_train_prepared)
y_pred = final_model.predict(X_test_prepared)
y_test_survived_pred = final_model.predict(X_test_survived_prepared)
print("Best accuracy:", grid.best_score_)
print("Train accuracy: ", accuracy_score(y_train,y_train_pred))
print("Final accuracy: ", accuracy_score(y_test_survived,y_test_survived_pred))
print("Final f1: ", f1_score(y_test_survived,y_test_survived_pred))
print("Final roc: ", roc_auc_score(y_test_survived,y_test_survived_pred))

# Submission

test_data["Survived"] = y_pred
result = test_data.loc[:,["PassengerId","Survived"]]
result.to_csv(os.path.join(here,"result.csv"),index = False)