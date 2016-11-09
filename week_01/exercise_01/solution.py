import pandas
import re

data = pandas.read_csv('../titanic.csv', index_col='PassengerId')

def formatted_float(x):
    return ("%0.2f" % float(x))

def percentage(subjects, count):
    return formatted_float(float(subjects) / count * 100)
def print_floats(numbers):
    print " ".join(map(lambda x: formatted_float(x), numbers))

# 1) sex counts
sex_counts = data['Sex'].value_counts()
male = sex_counts['male']
female = sex_counts['female']
print(" ".join([str(male), str(female)]))

# 2) survived percentage
survived_count = data['Survived'].value_counts()[1]
survived_percentage = percentage(survived_count, data['Sex'].size)
print survived_percentage

# 3) First class percentage
first_class = data['Pclass'].value_counts()[1]
print percentage(first_class, data['Pclass'].size)

# 4) mean and median of Age
median = data.Age.median()
mean = data.Age.mean()

print_floats([mean, median])
print(formatted_float(data.corr('pearson')['Parch']['SibSp']))

# most popular women name
women_names = data.loc[data.Sex == 'female'].Name

def name_if_no_bracket(string):
  return re.search(r'(?:Miss|Mrs)\.\s(\w+)', str(string))

def name_if_bracket(string):
  return re.search(r'\((\w+).*', string)

def parsed_name(string):
    if re.search(r'\(', string) != None:
        return name_if_bracket(string)
    else:
        return name_if_no_bracket(string)

def get_name(string):
    if parsed_name(string) != None:
        return parsed_name(string)
    else:
        return re.match(r'(.*)', string)

names_list = map(lambda x: str(x), women_names)

def filter_none(list):
    return filter(lambda x: x != None, list)

names = map(lambda x: x.group(1), filter_none(map(lambda x: get_name(str(x)), names_list)))

def name_reducer(acc, name):
    acc[name] = acc.get(name, 0) + 1
    return acc

names_dict = reduce(name_reducer, names, {})

import operator

sorted_dict = sorted(names_dict.items(), key=operator.itemgetter(1))


print 'Sorted dict: \n'
sorted_dict.reverse()
print(sorted_dict)
