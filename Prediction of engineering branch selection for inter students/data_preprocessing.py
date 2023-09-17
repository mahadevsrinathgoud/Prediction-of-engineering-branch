import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))

dataset = pd.read_csv("CollegeDataset/Dataset.csv", usecols=['rank', 'gender', 'caste', 'region', 'branch', 'college'], nrows=2000)
dataset.fillna(0, inplace=True)

encoder = []
columns = ['gender', 'caste', 'region', 'branch', 'college']

for i in range(len(columns)):
    le = LabelEncoder()
    dataset[columns[i]] = pd.Series(le.fit_transform(dataset[columns[i]].astype(str)))
    encoder.append(le)

dataset = dataset.values
X = dataset[:, 0:dataset.shape[1] - 1]
Y = dataset[:, dataset.shape[1] - 1]

X = sc.fit_transform(X)

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
Y = Y[indices]

# Save the processed data if needed
np.savez("processed_data.npz", X=X, Y=Y, encoder=encoder, sc=sc)
