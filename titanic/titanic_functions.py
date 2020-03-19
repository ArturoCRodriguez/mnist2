
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

def plot_learning_curves(model, pipeline, X_train, X_val, y_train, y_val):    
    train_errors, val_errors = [], []
    for m in range(1, len(X_train), 10):
        print("m= ",m)
        X_train_prepared = pipeline.fit_transform(X_train[:m])
        model.fit(X_train_prepared, y_train[:m])
        y_train_predict = model.predict(X_train_prepared)
        X_val_prepared = pipeline.transform(X_val)
        y_val_predict = model.predict(X_val_prepared)
        train_errors.append(f1_score(y_train[:m],y_train_predict))
        val_errors.append(f1_score(y_val,y_val_predict))
    plt.plot(np.sqrt(train_errors),"r-+",linewidth=2,label="train")
    plt.plot(np.sqrt(val_errors),"b-",linewidth=3,label="val")