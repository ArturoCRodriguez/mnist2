
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.base import BaseEstimator, TransformerMixin

class AgeBins(BaseEstimator, TransformerMixin):
    def __init__(self, ages = [10,20,40,60,120]):
        self.ages = ages
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):
        ages_array = np.array(self.ages)
        X["age_range"] = X["Age"].apply(lambda x: "age_"+str(ages_array[np.argmax(ages_array > x)]))
        X.drop("Age",axis=1, inplace=True)
        return X    
class CalculateFamilySize(BaseEstimator, TransformerMixin):
    def fit(self, X, y= None):
        return self
    def transform(self, X, y= None):        
        X["FamilySize"] = 1 + X["SibSp"] + X["Parch"]
        # X.drop(["SibSp","Parch"],axis= 1, inplace= True)
        X["IsAlone"] = X["FamilySize"].apply(lambda x: 1 if x > 0 else 1)
        # X.drop(["FamilySize"],axis= 1, inplace= True)
        return X
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
        # X["num_of_cabins"] = X.apply(get_num_of_cabins, axis= 1)
        # X["real_fare"] = X.apply(get_real_fare, axis= 1)
        # X["deck"] = X.apply(get_deck, axis= 1)
        X = X.drop(["Cabin","Fare"], axis= 1)
        return X
class GetDeck(BaseEstimator,TransformerMixin):    
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):                
        X["deck"] = X.apply(get_deck, axis= 1)
        X["HasCabin"] = X["Cabin"].apply(lambda c: 0 if c is np.nan or c is None else 1)
        X = X.drop(["Cabin"], axis= 1)
        return X
class Combine(BaseEstimator,TransformerMixin):    
    def fit(self, X,y= None):
        return self
    def transform(self, X, y= None):                
        X["Sex_ticket"] = X["Sex"] + "_" + X["Ticket"]
        X = X.drop(["Sex","Ticket"], axis= 1)
        return X
def plot_learning_curves(model, pipeline, X_train, X_val, y_train, y_val):    
    train_errors, val_errors = [], []
    for m in range(10, len(X_train), 10):
        print("m= ",m)
        X_train_prepared = pipeline.fit_transform(X_train[:m])
        model.fit(X_train_prepared, y_train[:m])
        y_train_predict = model.predict(X_train_prepared)
        X_val_prepared = pipeline.transform(X_val)
        y_val_predict = model.predict(X_val_prepared)
        train_errors.append(log_loss(y_train[:m],y_train_predict, labels=[0,1]))
        val_errors.append(log_loss(y_val,y_val_predict, labels=[0,1]))
    plt.plot(train_errors,"r-+",linewidth=2,label="train")
    plt.plot(val_errors,"b-",linewidth=3,label="val")

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
        return 'G'
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
        return 'G'
def print_results(model_name,y_true, y_predict):
    print("{} results:".format(model_name))
    # auc = roc_auc_score(y_true,y_predict)
    # f1 = f1_score(y_true,y_predict)
    # precision = precision_score(y_true,y_predict)
    # recall = recall_score(y_true,y_predict)
    # accuracy = accuracy_score(y_true,y_predict)
    # print("auc\t f1\t precision\t recall\t accuracy")
    # print("{}\t {}\t {}\t {}\t {}".format(auc,f1,precision,recall,accuracy))
    accuracy = accuracy_score(y_true,y_predict)
    log_loss_error = log_loss(y_true,y_predict, labels=[0,1])
    print("accuracy\t log_loss")
    print("{}\t {}".format(accuracy, log_loss_error))

