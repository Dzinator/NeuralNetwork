import numpy as np
import pandas as pd
import csv

from sklearn import linear_model


x = pd.read_csv("data/train_x.csv", delimiter=",").values
y = pd.read_csv("data/train_y.csv", delimiter=",").values
test_x = pd.read_csv("data/test_x.csv", delimiter=",").values

# Normalize data
x = x / 255.0
# y = y.reshape(-1, 1)
test_x = test_x / 255.0

train_data_size = len(x)

# Logistic Regression

clf = linear_model.LogisticRegressionCV()
clf.fit(x, y)

# Predict and save

results = clf.predict(test_x)

with open('predicted.csv', 'w') as f:
    writer = csv.writer(f)
    for r in results:
        writer.writerow(r)

print results[:10]
