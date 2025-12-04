import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

train = pd.read_csv("labelled1.csv")

# Distribution of normal and anomaly traffic in train data
print(train['class'].value_counts())

# PREPROCESSING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
# extract numerical attributes and scale it to have zero mean and unit variance
cols = train.select_dtypes(include=['float64', 'int64']).columns
sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
# turn the result back to a dataframe
train_x = pd.DataFrame(sc_train, columns=cols)
train_y = train['class']
print(train_x.shape)
print(train_y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, train_size=0.70, random_state=1, stratify=train_y)
df_analysis = pd.concat([X_test, Y_test], axis=1)

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import metrics
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

# Create a svm Classifier
clf1 = OneVsOneClassifier(svm.SVC(kernel='linear'))  # Linear Kernel
clf1.fit(X_train, Y_train)

# RandomForest classifier
clf2 = OneVsRestClassifier(RandomForestClassifier(random_state=1))
clf2.fit(X_train, Y_train)

# KNN classifer
clf3 = OneVsRestClassifier(KNeighborsClassifier())
clf3.fit(X_train, Y_train)

# LogisticRegression
clf4 = OneVsRestClassifier(LogisticRegression(multi_class='multinomial', max_iter=1000))
clf4.fit(X_train, Y_train)

# DTC
clf5 = OneVsRestClassifier(DecisionTreeClassifier())
clf5.fit(X_train, Y_train)

# Naive Bayes
clf6 = OneVsRestClassifier(GaussianNB())
clf6.fit(X_train, Y_train)

# ANN
clf7 = OneVsRestClassifier(MLPClassifier(solver='lbfgs', max_iter=10000))
clf7.fit(X_train, Y_train)

# GradientBoosting
from sklearn.ensemble import GradientBoostingClassifier

clf11 = OneVsRestClassifier(GradientBoostingClassifier())
clf11.fit(X_train, Y_train)

# Stacking
clf12 = OneVsRestClassifier(StackingClassifier(
    estimators=[('dtc', DecisionTreeClassifier()), ('gb', GradientBoostingClassifier()), ('gnb', GaussianNB())],
    final_estimator=MLPClassifier(solver='lbfgs', max_iter=10000), cv=10))
clf12.fit(X_train, Y_train)

models = []
models.append(('SVM', clf1))
models.append(('RandomForest', clf2))
models.append(('KNN', clf3))
models.append(('LogisticRegression', clf4))
models.append(('DTC', clf5))
models.append(('NB', clf6))
models.append(('MLP', clf7))
models.append(('GB', clf11))
models.append(('Stacking', clf12))

model_test_accuracy = []
model_test_precision = []
model_test_f1 = []
model_test_recall = []

for i, v in models:
    start = time.time()
    v.fit(X_train, Y_train)
    stop = time.time()
    print("Training time: ", stop - start)
    start = time.time()
    Y_pred = v.predict(X_test)
    stop = time.time()
    print("Prediction time:", stop - start)
    accuracy = metrics.accuracy_score(Y_test, v.predict(X_test))
    confusion_matrix = metrics.confusion_matrix(Y_test, v.predict(X_test))
    classification = metrics.classification_report(Y_test, v.predict(X_test), output_dict=True)
    print()
    print('============================== {} Model Test Results =============================='.format(i))
    print()
    print("Model Accuracy:" "\n", accuracy)
    print()
    print("Confusion matrix:" "\n", confusion_matrix)
    print()
    print("Classification report:" "\n", classification)
    print()
    ml_cfm = metrics.multilabel_confusion_matrix(Y_test, v.predict(X_test))
    model_test_accuracy.append(accuracy)
    model_test_precision.append(classification['macro avg']['precision'])
    model_test_f1.append(classification['macro avg']['f1-score'])
    model_test_recall.append(classification['macro avg']['recall'])

    ConfusionMatrixDisplay.from_estimator(v, X_test, Y_test, cmap='Blues', xticks_rotation='vertical', normalize='true',
                          display_labels=['CMM', 'CI', 'NMM', 'NDoS', 'Normal'])
    # metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_perc, display_labels=label_classes).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.show()

model_names = ['SVM', 'RF', 'KNN', 'LR', 'DTC', 'NB', 'MLP', 'GB', 'Stacking']

fig, axes = plt.subplots(figsize=(8, 8))
f1 = axes.plot(model_names, model_test_accuracy, marker='s', color='#069AF3')
plt.title('Accuracy')
axes.set_xlabel('Algorithms', fontsize=16)
axes.set_ylabel('Accuracy_score', fontsize=16)
axes.grid(True)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
axes.plot(model_names, model_test_precision, marker='s', color='g')
# axes.set_ylim(ymin=0)
font_size = 15
plt.title('Precision')
plt.xlabel('Algorithms', fontsize=16)
plt.ylabel('Precision', fontsize=16)
axes.grid(True)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
axes.plot(model_names, model_test_recall, marker='s', color='orange')
# axes.set_ylim(ymin=0)
font_size = 15
plt.title('Recall')
plt.xlabel('Algorithms', fontsize=16)
plt.ylabel('Recall', fontsize=16)
axes.grid(True)
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
axes.plot(model_names, model_test_f1, marker='s', color='teal')
# axes.set_ylim(ymin=0)
font_size = 15
plt.title('F1_score')
plt.xlabel('Algorithms', fontsize=16)
plt.ylabel('F1-score', fontsize=16)
axes.grid(True)
plt.show()