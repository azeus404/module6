# Load libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from collections import Counter


import argparse


parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')


args = parser.parse_args()
path = args.path

"""
 loading data
"""
df = pd.read_csv(path, encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("[*] Adding features")
"""
Shannon Entropy calulation
"""
def calcEntropy(x):
    p, lens = Counter(x), np.float(len(x))
    return -sum( count/lens * np.log2(count/lens) for count in p.values())

df['entropy'] = [calcEntropy(x) for x in df['lld']]

"""
LLD record length
"""
df['length'] = [len(x) for x in df['lld']]


"""

 Number of different characters

"""
def countChar(x):
    charsum = 0
    total = len(x)
    for char in x:
        if not char.isalpha():
            charsum = charsum + 1
    return float(charsum)/total
df['numbchars'] = [countChar(x) for x in df['lld']]

"""
Number of . in subdomain
"""
df['numbdots'] = [x.count('.') for x in df['lld']]

"""
Number of character in subdomain
"""
df['numunique'] = [len(set(x)) for x in df['lld']]


"""

    Build models

"""
models = []
models.append(('NN',MLPClassifier()))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('RF', RandomForestClassifier(n_estimators=10)))

#
x = df.drop(['label','lld'], axis=1)
y = df['label']
X_train, X_validation, Y_train, Y_validation = train_test_split(x, y, test_size=0.20, random_state=1)


# evaluate each model in turn scored on accuracy = correct predictions/total predictions
results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
