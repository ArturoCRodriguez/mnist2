import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, f1_score, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal

mnist = pd.read_csv("mnist_784_csv.csv")
X,y = mnist.drop("class",axis=1).to_numpy(), mnist["class"].copy().values
y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

some_digit = X[2,:]

sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train,y_train)
# # print(some_digit.shape)
# print(sgd_clf.predict([some_digit]))
# print(y_train[2])
# print(sgd_clf.decision_function([some_digit]))

# forest_clf = RandomForestClassifier(random_state= 42)
# forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv= 3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(y_train_pred)
print(conf_mx)