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

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal

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
# print("F1 Score: ", f1_score(y_train_5,y_train_pred))
y_scores = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3, method="decision_function",n_jobs= 8)
# pd.set_option('display.max_rows', None)
# np.set_printoptions(threshold=9999999)

# print(y_scores)
# precisions, recalls, thresholds = precision_recall_curve(y_train_5,y_scores)
# print(thresholds)
# plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
# plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
# plt.plot(recalls, precisions, linewidth= 2, label= None)
# plt.plot([0,1], [0,1], 'k--')
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)
# plt.plot(fpr,tpr,linewidth=2,label= None)
# plt.plot([0,1],[0,1],'k--')
# plt.show()
# print(roc_auc_score(y_train_5, y_scores))

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf,X_train, y_train_5, cv=3,method="predict_proba", n_jobs=8)
y_pred = cross_val_predict(forest_clf,X_train, y_train_5, cv=3, method="predict",n_jobs= 4)

print(y_probas_forest)
y_scores_forest = y_probas_forest[:,1]
fpr_forest, tpr_forest, threshold_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc= "lower right")
plt.show()

print(roc_auc_score(y_train_5,y_scores_forest))
print(precision_score(y_train_5, y_pred))
print(recall_score(y_train_5, y_pred))