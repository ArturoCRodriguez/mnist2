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
mnist = pd.read_csv("mnist_784_csv.csv")
X,y = mnist.drop("class",axis=1).to_numpy(), mnist["class"].copy().values
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train,y_train_5)
# result = sgd_clf.predict(X_test)
# accuracy = (result == y_test_5)
# print(np.sum(accuracy == True)/y_test_5.shape[0])
# Devuelve la puntiación de la métrica de scoring para cada conjunto CV
result = cross_val_score(sgd_clf,X_train, y_train_5,cv=3, scoring="recall")
print(result)
# Devuelve valores para cada elemento de X_train en función de method
# "decision_function" hace que se devuelvan los valores de decisión calculados por el modelo
# "predict" hace que se devuelvan los valores de la clase que se predice (en este caso True o False, que son las clases de y_train_5)
y_scores = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3, method="decision_function",n_jobs= 4)
print(y_scores)

# y_train_pred = cross_val_predict(sgd_clf,X_train, y_train_5,cv=3)
# print(confusion_matrix(y_train_5,y_train_pred))
# print("Precision: ",precision_score(y_train_5,y_train_pred))
# print("Recall: ", recall_score(y_train_5,y_train_pred))
# print("F1 Score: ", f1_score(y_train_5,y_train_pred))
# y_scores = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3, method="decision_function",n_jobs= 4)
# pd.set_option('display.max_rows', None)
# np.set_printoptions(threshold=9999999)

# print(y_scores)
# precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
# print(thresholds)
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# plt.plot(recalls, precisions, linewidth= 2, label= None)
# plt.plot([0,1], [0,1], 'k--')
# fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# plt.plot(fpr,tpr,linewidth=2,label= None)
# plt.plot([0,1],[0,1],'k--')
# plt.show()
# print(roc_auc_score(y_train_5, y_scores))