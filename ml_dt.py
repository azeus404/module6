import pandas as pd
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
from warnings import filterwarnings
filterwarnings('ignore')

import pydotplus
import argparse

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score
from sklearn.tree import DecisionTreeClassifier


#graph
from pydot import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')
args = parser.parse_args()
path = args.path

""""
Pre-process data: drop duplicates and empty
"""
df = pd.read_csv(path,encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)


print(df.head)
"""
Properties of the dataset
"""
#print('%d %d' % (df.shape[0], df.shape[1]))
#print('Total domains %d' % df.shape[0])

"""
Simple Decison Tree
"""

dt=DecisionTreeClassifier()

#features, labels
x = df.drop(['label','lld'], axis=1)
y = df['label']

# Training set size is 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
dt.fit(x_train,y_train)

print("Accuracy score: ",dt.score(x_test,y_test))


"""
Performance
- Confusion matrix
- Classification report
- ROC
"""
y_pred = dt.predict(x_test)
y_true = y_test

print("[+]Confusion matrix")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

print("[+]classification report")
print(classification_report(y_test, y_pred))

y_pred_proba = dt.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Decision Tree ROC curve')
plt.show()
print('Area under the ROC Curve %d' % roc_auc_score(y_test,y_pred_proba))
#http://gim.unmc.edu/dxtests/ROC3.htm
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")

"""
Cross validation k-fold
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=42)
model_kfold = DecisionTreeClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
