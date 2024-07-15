import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

datasize = 10000000000000

ydf = np.asarray(pd.read_csv("Fuzzing_labels.csv", nrows = datasize), dtype=np.int32)[1:, 1]
xdf = np.asarray(pd.read_csv("Fuzzing_dataset.csv", nrows = datasize), dtype = np.float32)
print(xdf.shape, ydf.shape)




X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size = 0.1, random_state=1)
clf = RandomForestClassifier(n_estimators = 21, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
t1 = clf.predict(X_test) 
t2 = y_test
np.save("preds.npy", t1)
np.save("truth.npy", t2)
print(np.mean(np.abs(t1 - t2)))
print(np.mean(y_test))