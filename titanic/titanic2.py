#%%
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.preprocessing import OneHotEncoder,RobustScaler, Normalizer, OrdinalEncoder
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
here = os.path.dirname(os.path.abspath(__file__))

train_data = pd.read_csv( os.path.join(here,"train.csv"))
test_data = pd.read_csv( os.path.join(here,"test.csv"))
test_data_survived = pd.read_csv( os.path.join(here,"test_survived.csv"))

gender_data = pd.read_csv( os.path.join(here,"gender_submission.csv"))
# print(train_data[train_data["Embarked"].isnull()])
pd.set_option('display.max_rows', None)
# print(train_data[train_data["Cabin"].notnull()])
print(train_data.info())
# train_data.loc[:,["Ticket","Fare","Cabin","Pclass", "Name","Survived"]].sort_values(['Fare'], ascending=False)
train_data.sort_values(['Age'], ascending=False)


#%%
columns_to_drop = ["PassengerId","Name","Embarked","Ticket"]
num_columns = ["SibSp","Parch","Pclass","Fare"]
num_columns_to_transform = ["Age"]
cat_columns_stay = ["Sex"]
cat_columns_to_transform = ["Cabin"]

num_pipeline = Pipeline(
    [        
        ('family', CalculateFamilySize()),
        ('imputer',  SimpleImputer(strategy="median")),
        ('std_scaler',RobustScaler(1,99)),   
        # ('bins',KBinsDiscretizer(n_bins=4, encode='onehot-dense',strategy='uniform'))    
    ]
)
num_to_transform_pipeline = Pipeline(
    [
        ('bins',AgeBins()),
        ('bins_ohe',OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
cat_stay_pipeline = Pipeline(
    [          
        ('imputer_cat', SimpleImputer( strategy="most_frequent")),        
        ("cat3",OneHotEncoder(handle_unknown="ignore", sparse= False))
    ]
)
cat_transform_pipeline = Pipeline(
    [   ('cabin', GetDeck()),     
        # ('decks', Combine()),        
        # ("decks2",OneHotEncoder(handle_unknown="ignore", sparse= False))        
        # ("decks2",OrdinalEncoder())        
    ]
)
transformer = ColumnTransformer([
    ("num",num_pipeline,num_columns),
    ("num_transform",num_to_transform_pipeline,num_columns_to_transform),
    ("cat_stay",cat_stay_pipeline,cat_columns_stay),
    ("cat_transform",cat_transform_pipeline, cat_columns_to_transform)
])

full_pipeline = Pipeline([
    ("1",DropColumns(columns_to_drop)),
    ("2",transformer)
]) 
# train_data.drop(["PassengerId","Name"], axis= 1, inplace= True)
#%%
# split = StratifiedShuffleSplit(n_splits=1,test_size=0.3,random_state=42)
# for train_index, test_index in split.split(train_data,train_data["Sex"]):
#     train_data_train = train_data.iloc[train_index]
#     val_data = train_data.iloc[test_index]

X_train_data = train_data.drop("Survived", axis=1)
y_train_data = train_data["Survived"]
X_train, X_val, y_train, y_val = train_test_split(X_train_data, y_train_data, random_state=42, test_size = 0.1)
#%%

# X_train = train_data_train.drop("Survived", axis=1)
# y_train = train_data_train["Survived"]
# X_train = train_data.drop("Survived", axis=1)
# y_train = train_data["Survived"]
# X_val = val_data.drop("Survived", axis=1)
# y_val = val_data["Survived"]

X_train_prepared = full_pipeline.fit_transform(X_train)
X_val_prepared = full_pipeline.transform(X_val)

#%%
clf = RandomForestClassifier(random_state=42, n_jobs=4)
clf2 = LGBMClassifier(n_jobs=4)
clf3 = MLPClassifier(random_state=42)
clf4 = AdaBoostClassifier(random_state=42)
clf5 = xgb.XGBClassifier(random_state=42)
params = [{'n_estimators':[10,20,30,40,50,70,100,250,260,270,280,290,300,320,350,400, 500, 510]}]
# params = [{'n_estimators':[500,510,530,550,570,600]}]
# params = [{'max_iter':[100,200,270,300,400]}]
# params = [
#     {'solver': ["lbfgs","adam"],'activation':["relu","logistic"], 'max_iter':[100,200,270,300,400], 'alpha': [0.01,0.03,0.1,0.3], 'learning_rate_init':[0.01,0.03,0.1,0.3]}
# ]
grid = GridSearchCV(estimator = clf, param_grid= params, scoring='accuracy',n_jobs=4,cv=12, verbose=100)
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
# final_model.fit(X_train_prepared,y_train)
y_pred2 = final_model.predict(X_val_prepared)
# y_pred3 = clf3.predict(X_val_prepared)

y_train_pred2 = final_model.predict(X_train_prepared)


# print_results("Forest",y_val, y_pred)
# print_results("LightGBM Train",y_train, y_train_pred2)
print_results("Validation accuracy: ",y_val, y_pred2)
# print_results("NN",y_val, y_pred3)

# Submission
X_test_prepared = full_pipeline.transform(test_data)
y_sub = final_model.predict(X_test_prepared)
test_data["Survived"] = y_sub
result = test_data.loc[:,["PassengerId","Survived"]]
result.to_csv(os.path.join(here,"result.csv"),index = False)

# Scoring
X_test_survived = test_data_survived.drop("Survived", axis=1)
y_test_survived = test_data_survived["Survived"]
X_test_survived_prepared = full_pipeline.transform(X_test_survived)
y_test_survived_pred = final_model.predict(X_test_survived_prepared)
print_results("Test accuracy: ",y_test_survived, y_test_survived_pred)
print("Shape: ", X_train_prepared.shape)
