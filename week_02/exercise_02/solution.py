import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import *
from sklearn.preprocessing import *
from sklearn import metrics
import operator

def print_results(results):
  sorted_results = sorted(results.items(), key=operator.itemgetter(0))
  print sorted_results[0]
  print sorted_results[len(sorted_results)-1]

def run_estimations(v_range, X, y, cv):
  means = dict()
  for i in v_range:
    kn = KNeighborsRegressor(n_neighbors=5, p=i, metric='minkowski', weights='distance')
    kn.fit(X, y)
    array = cross_val_score(estimator=kn, X=X, y=y, cv=cv, scoring='neg_mean_squared_error')
    print i
    m = array.mean()
    print m
    means[i] = m
  return means

raw = load_boston()

X = raw.data
y = raw.target

X_scaled = scale(X)

kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)
p_range = np.linspace(1, 10, 200)

kmeans = run_estimations(p_range, X_scaled, y, kf)

print_results(kmeans)
