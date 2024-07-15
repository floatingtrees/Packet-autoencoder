import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.tree import export_graphviz
import os
import graphviz
from sklearn.ensemble import RandomForestClassifier
datasize = 1000000000000000000000

xdf = np.asarray(pd.read_csv("fixed_hurst.csv", nrows = datasize, header=None), dtype=np.float32)[1:2472400, :]
thingy = np.asarray(pd.read_csv("Fuzzing_dataset.csv", nrows = datasize, dtype = np.float32))[1:2472400, :]
#print(xdf.shape, thingy.shape);exit()
#xdf = np.concatenate((xdf, thingy), axis = 1)

ydf = np.asarray(pd.read_csv("Video_Injection_labels.csv", nrows = datasize), dtype = np.int32)[1:2472400, :].squeeze()


X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size = 0.1, random_state=1)
clf = RandomForestClassifier(n_estimators = 21, max_depth=5, random_state=42)
clf.fit(X_train, y_train)

t1 = clf.predict(X_test) 
t2 = y_test
print(np.mean(np.abs(t1 - t2)))
print(np.mean(y_train))
preds = t1
truth = t2

precision = precision_score(truth, preds, average=None)
recall = recall_score(truth, preds, average=None)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {2 * precision * recall / (precision + recall)}")

