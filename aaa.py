import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, precision_score, recall_score

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
datasize = 100000000000000
df = pd.read_csv("MachineLearningCVE/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv", nrows = datasize)
df = df.replace([np.inf, -np.inf], np.nan)

df_last_column = df.iloc[:, -1]  # DataFrame with only the last column
df_other_columns = df.iloc[:, :-1] 
xdf = np.asarray((df_other_columns), dtype=np.float32)
ydf = np.empty(df.shape[0], dtype = np.int32)
for i in range(ydf.shape[0]):
    if df_last_column[i] == "BENIGN":
        ydf[i] = 0
    elif df_last_column[i] == "PortScan":
        ydf[i] = 1
    else:
        raise ValueError(f"{df_last_column[i]}")



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


X_train, X_test, y_train, y_test = train_test_split(xdf, ydf, test_size = 0.1, random_state=1)
clf = RandomForestClassifier(n_estimators = 21, max_depth=5, random_state=42)
clf.fit(X_train, y_train)
t1 = clf.predict(X_test) 
t2 = y_test
preds = t1
truth = t2

precision = precision_score(truth, preds)
recall = recall_score(truth, preds)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {2 * precision * recall / (precision + recall)}")