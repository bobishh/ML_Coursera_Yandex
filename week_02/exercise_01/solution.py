import pandas
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import *
from sklearn.preprocessing import *
from sklearn import metrics

def run_estimations(v_range, X, y, cv):
  means = list()
  for i in v_range:
    kn = KNeighborsClassifier(n_neighbors=i)
    kn.fit(X, y)
    array = cross_val_score(estimator=kn, X=X, y=y, cv=cv, scoring='accuracy')
    m = array.mean()
    means.append(m)
  return means

def print_results(results):
  m = max(results)
  indices = [i for i, j in enumerate(results) if j == m]
  print indices[0]+1
  print np.round(m, decimals=2)

data = pandas.read_csv('wine.data', header=None)
X = pandas.read_csv('wine.data', header=None, usecols=list(xrange(1,14)))
y = pandas.read_csv('wine.data', header=None, usecols=[0]).values.reshape(len(X),)

kf = KFold(n=len(X), n_folds=5, shuffle=True, random_state=42)

kMeans1 = run_estimations(range(1, 51), X, y, kf)
print_results(kMeans1)

X_scale = scale(X)
kMeans2 = run_estimations(range(1, 51), X_scale, y, kf)
print_results(kMeans2)
