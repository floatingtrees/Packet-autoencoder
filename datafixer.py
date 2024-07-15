import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score


preds = np.load("preds.npy")
truth = np.load("truth.npy")

precision = precision_score(truth, preds)
recall = recall_score(truth, preds)
print(precision)
print(recall)
print(2 * precision * recall / (precision + recall))