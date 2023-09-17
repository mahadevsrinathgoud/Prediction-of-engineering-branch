import numpy as np
import pandas as pd

# Load the processed data
data = np.load("processed_data.npz")
encoder = data['encoder']
sc = data['sc']

testData = [71654, 'F', 'BC_B', 'OU', 'PHARM - D (M.P.C. STREAM)']
temp = []
temp.append(testData)
temp = np.asarray(temp)

df = pd.DataFrame(temp, columns=['rank', 'gender', 'caste', 'region', 'branch'])
for i in range(len(encoder) - 1):
    df[columns[i]] = pd.Series(encoder[i].transform(df[columns[i]].astype(str)))

df = df.values
df = sc.transform(df)

predict = cls.predict(df)
print(predict)
print(encoder[4].inverse_transform(predict))
