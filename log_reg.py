import numpy as np
import pandas as pd
from sklearn import linear_model


x = pd.read_csv("data/train_x.csv", delimiter=",", memory_map=True).values
y = pd.read_csv("data/train_y.csv", delimiter=",", memory_map=True).values
test_x = pd.read_csv("data/test_x.csv", delimiter=",", memory_map=True).values

# Normalize data
x = x / 255.0
# y = y.reshape(1, -1)
test_x = test_x / 255.0

train_data_size = len(x)

# Logistic Regression

clf = linear_model.LogisticRegressionCV()
clf.fit(x, list(y))

# Predict and save

results = clf.predict(test_x)

np.savetxt('predicted.csv', results, delimiter=',')

print(results[:10])
