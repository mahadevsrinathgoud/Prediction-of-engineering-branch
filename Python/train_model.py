import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the processed data
data = np.load("processed_data.npz")
X = data['X']
Y = data['Y']
encoder = data['encoder']
sc = data['sc']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

cls = RandomForestClassifier()
cls.fit(X_train, y_train)
predict = cls.predict(X_test)
a = accuracy_score(y_test, predict) * 100
print(a)
