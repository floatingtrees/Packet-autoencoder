import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

datasize = 10000000000000

xdf = np.asarray(pd.read_csv("Video_Injection_dataset.csv", nrows = datasize), dtype=np.float32)
ydf = np.asarray(pd.read_csv("Video_Injection_labels.csv", nrows = datasize), dtype = np.int32)[1:, 1]
print(xdf.shape, ydf.shape)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Example data: 10 samples with 5 features each
data = xdf

# Example labels: 10 samples
labels = ydf

# Perform PCA
pca = PCA(n_components=2, random_state = 42)
principal_components = pca.fit_transform(data)

# Assign colors based on labels
colors = ['red' if label == 1 else 'blue' for label in labels]

# Plotting the PCA
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=colors, marker='o')

# Adding labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Features')

# Show plot
plt.grid()
plt.show()

exit()
from sklearn.metrics import confusion_matrix, precision_score, recall_score

X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size = 0.1, random_state=1)
clf = RandomForestClassifier(n_estimators = 21, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
t1 = clf.predict(X_test) 
t2 = y_test
np.save("preds.npy", t1)
np.save("truth.npy", t2)
print(np.mean(np.abs(t1 - t2)))
print(np.mean(y_test))
preds = t1
truth = t2

precision = precision_score(truth, preds, average=None)
recall = recall_score(truth, preds, average=None)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {2 * precision * recall / (precision + recall)}")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Example data: 10 samples with 5 features each
data = X_test

# Example labels: 10 samples
labels1 = preds
labels2 = truth

# Perform PCA
pca = PCA(n_components=2, random_state = 42)
principal_components = pca.fit_transform(data)

# Assign colors based on labels
colors = ['red' if (labels1[i] == 0 and labels2[i] == 1) else 'blue' for i in range(len(labels1))]

# Plotting the PCA
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=colors, marker='o')

# Adding labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Features')

# Show plot
plt.grid()
plt.show()