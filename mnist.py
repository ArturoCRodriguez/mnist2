import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier  
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score, confusion_matrix, precision_recall_curve
mnist = pd.read_csv("mnist_784_csv.csv")
# print(mnist.head())
# mnist = fetch_openml("mnist_784",version=1)
# print(mnist.keys())

X,y = mnist.drop("class",axis=1).to_numpy(), mnist["class"].copy().values
# X,y = mnist["data"], mnist["target"]

y = y.astype(np.uint8)
# print(y[0])
# some_digit = X.iloc[0,:]
# some_digit_image = some_digit.values.reshape(28,28)
# plt.imshow(some_digit_image, cmap = mpl.cm.get_cmap("binary"), interpolation="nearest") 
# plt.axis("off") 
# plt.show()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
# result = sgd_clf.predict(X_test)
# accuracy = (result == y_test_5)
# print(np.sum(accuracy == True)/y_test_5.shape[0])
# result = cross_val_score(sgd_clf,X_train, y_train_5,cv=3, scoring="accuracy")
# y_train_pred = cross_val_predict(sgd_clf,X_train, y_train_5,cv=3)
# print(confusion_matrix(y_train_5,y_train_pred))
# print("Precision: ",precision_score(y_train_5,y_train_pred))
# print("Recall: ", recall_score(y_train_5,y_train_pred))
y_scores = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3, method="decision_function")
print(y_scores)
precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.show()