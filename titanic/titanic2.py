#%%
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder,RobustScaler, Normalizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from titanic_functions import plot_learning_curves,get_num_of_cabins,get_real_fare,get_deck,DropColumns,AddFeaturesFromCat,GetDeck, print_results,Combine

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
columns_to_drop = ["PassengerId","Name","Cabin","Embarked"]
num_columns = ["Age","SibSp","Parch","Fare","Pclass"]
cat_columns_stay = ["Sex","Ticket"]
cat_columns_to_transform = ["Sex","Ticket"]

num_pipeline = Pipeline(
    [
        # ("columns",DropColumns(cat_columns)),
        ('imputer',  SimpleImputer(strategy="median")),
        ('std_scaler',RobustScaler(1,99)),        
    ]
)
cat_stay_pipeline = Pipeline(
    [          
        ('imputer_cat', SimpleImputer( strategy="most_frequent")),        
        ("cat3",OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
cat_transform_pipeline = Pipeline(
    [        
        ('decks', Combine()),        
        ("decks2",OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
transformer = ColumnTransformer([
    ("num",num_pipeline,num_columns),
    ("cat_stay",cat_stay_pipeline,cat_columns_stay),
    # ("cat_transform",cat_transform_pipeline, cat_columns_to_transform)
])

full_pipeline = Pipeline([
    ("1",DropColumns(columns_to_drop)),
    ("2",transformer)
]) 
# train_data.drop(["PassengerId","Name"], axis= 1, inplace= True)
#%%
split = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=42)
for train_index, test_index in split.split(train_data,train_data["Sex"]):
    train_data_train = train_data.iloc[train_index]
    val_data = train_data.iloc[test_index]
#%%

# X_train = train_data_train.drop("Survived", axis=1)
# y_train = train_data_train["Survived"]
X_train = train_data.drop("Survived", axis=1)
y_train = train_data["Survived"]
X_val = val_data.drop("Survived", axis=1)
y_val = val_data["Survived"]

X_train_prepared = full_pipeline.fit_transform(X_train)
X_val_prepared = full_pipeline.transform(X_val)

#%%
clf = RandomForestClassifier(random_state=42, n_jobs=4)
clf2 = LGBMClassifier(n_jobs=4)
clf3 = MLPClassifier(random_state=42,solver='lbfgs')
params = [{'n_estimators':[10,20,30,40,50,70,100,250,260,270,280,290,300,320,350,400]}]
# params = [{'max_iter':[100,200,270,300,400]}]
grid = GridSearchCV(estimator = clf, param_grid= params, scoring='accuracy',n_jobs=4,cv=5, verbose=100)
grid.fit(X_train_prepared,y_train)
cvres = grid.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(mean_score, params)
print("Best:", grid.best_params_)
final_model = grid.best_estimator_
# plot_learning_curves(clf,full_pipeline,X_train,X_val,y_train,y_val)
# plt.show()
#%%
# clf.fit(X_train_prepared, y_train)
# clf2.fit(X_train_prepared,y_train)
# clf3.fit(X_train_prepared, y_train)

# y_pred = clf.predict(X_val_prepared)
y_pred2 = final_model.predict(X_val_prepared)
# y_pred3 = clf3.predict(X_val_prepared)

y_train_pred2 = final_model.predict(X_train_prepared)


# print_results("Forest",y_val, y_pred)
# print_results("LightGBM Train",y_train, y_train_pred2)
# print_results("LightGBM",y_val, y_pred2)
# print_results("NN",y_val, y_pred3)

# Submission
X_test_prepared = full_pipeline.transform(test_data)
y_sub = final_model.predict(X_test_prepared)
test_data["Survived"] = y_sub
result = test_data.loc[:,["PassengerId","Survived"]]
result.to_csv(os.path.join(here,"result.csv"),index = False)

print("Shape: ", X_train_prepared.shape)