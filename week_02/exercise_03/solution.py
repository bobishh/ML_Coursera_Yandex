import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

import numpy as np
from sklearn.linear_model import Perceptron

def train_and_test(perceptron, X, y, X_test, y_test):
    perceptron.fit(X, y)
    return perceptron.score(X_test, y_test)

def map_to_val(arr):
    return map(lambda x: x[0], arr)

scaler = StandardScaler()

X_train = pandas.read_csv('train_data.csv', header=None, usecols=[1, 2]).values
X_test = pandas.read_csv('test_data.csv', header=None, usecols=[1, 2]).values

y_train = map_to_val(pandas.read_csv('train_data.csv', header=None, usecols=[0]).values)
y_test = map_to_val(pandas.read_csv('test_data.csv', header=None, usecols=[0]).values)

clf = Perceptron(random_state=241)
print("Learning on unscaled data accuracy")
score_unscaled = train_and_test(clf, X_train, y_train, X_test, y_test)
print(score_unscaled)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Learning on scaled data accuracy")
score_scaled = train_and_test(clf, X_train_scaled, y_train, X_test_scaled, y_test)
print(score_scaled)
score_diff = score_scaled - score_unscaled

print("Scores diff:")
print np.round(score_diff, decimals=3)
