import pandas
import re
import numpy as np
from sklearn.tree import DecisionTreeClassifier

data = pandas.read_csv('../titanic.csv', index_col='PassengerId')

def formatted_float(x):
  return ("%0.2f" % float(x))

def percentage(subjects, count):
  return formatted_float(float(subjects) / count * 100)
def print_floats(numbers):
  print " ".join(map(lambda x: formatted_float(x), numbers))


def map_to_int(list):
  return map(lambda x: int(x), list)

def map_to_float(list):
  return map(lambda x: float(x), list)

# 1 filter from where age isnan # other features are not null anyway
filtered = data.loc[data.Age.notnull()]

# 2 filter out redundant fields
# 2b make sex boolean field

features = filtered[['Age', 'Fare', 'Pclass', 'Sex', 'Survived']]

def map_sex(values):
  values[4] = True if values[4] == 'female' else False
  return values

features_records = features.to_records()
map(lambda x: map_sex(x), features_records)
features_values = map(lambda x: [x.Age, x.Sex, x.Pclass, x.Fare], features_records)
# 3 filter out target value (Survived)
results = map(lambda x: [x.Survived], features_records)
# 4 create classifier and learn it
X=np.array(features_values)
y=np.array(results)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X,y)

# 5 get important features
print "Age, Fare, Pclass, Sex importance: "
print clf.feature_importances_
