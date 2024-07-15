import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

datasize = 10000000000000

hursts = np.asarray(pd.read_csv("outputs.csv", nrows = datasize, header=None), dtype=np.float32)
ydf = np.asarray(pd.read_csv("Name_Time_Fuzzing", nrows = datasize, header=None), dtype = np.float32)
ydf = np.squeeze(ydf)
new_array = np.zeros((ydf.shape[0], hursts.shape[1]))
print(hursts.shape, ydf.shape)
for i in range(ydf.shape[0]):
    time = ydf[i]
    converted_time = int(time/0.01)
    if converted_time > 173690:
        continue
    new_array[i, :] = hursts[converted_time, :]

df = pd.DataFrame(new_array)
df.to_csv("fixed_hurst2.csv")
assert np.min(new_array) >= 0, np.min(new_array)
