import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

datasize = 100000000000000000

xdf = np.asarray(pd.read_csv("Fuzzing_dataset.csv", nrows = datasize), dtype=np.float32)
ydf = np.asarray(pd.read_csv("Fuzzing_labels.csv", nrows = datasize), dtype = np.int32)[1:, 1]

np.set_printoptions(precision = 5, suppress = True)

def PCA(data, dim = 2):
    np.set_printoptions(precision = 5, suppress = True)
    epsilon = 1e-8
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    norm = (data - mean) / (std + epsilon)
    covariances = np.cov(norm.T)
    values, vectors = np.linalg.eig(covariances)
    sorted_order = np.argsort(values)
    value_sum = np.sum(values)
    sorted_vectors = vectors[sorted_order][:, :dim].T
    relevant_values = values[:dim]
    fds = np.matmul(sorted_vectors, data.T)
    print(relevant_values/value_sum)
    return fds.T

# Example data: 10 samples with 5 features each

data = xdf
labels = ydf
selected_indices = np.random.choice(data.shape[0], data.shape[0], replace=False)

# Example labels: 10 samples

principal_components = PCA(data)
#principal_components = principal_components[selected_indices, :]
#labels = labels[selected_indices]
# Assign colors based on labels
colors = ['red' if label == 1 else 'blue' for label in labels]

print(np.mean(principal_components), np.std(principal_components))

# Plotting the PCA
plt.figure(figsize=(10, 6))
plt.scatter(principal_components[:,0], principal_components[:, 1], c=colors, marker='o')

# Adding labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Features')

# Show plot
plt.grid()
plt.show()