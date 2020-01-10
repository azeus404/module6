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

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.dummy import DummyClassifier

#graph
from pydot import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')
parser.add_argument('--print', help='print Decision Tree')

args = parser.parse_args()
path = args.path
print = args.print


""""
Pre-process data: drop duplicates and empty
"""
df = pd.read_csv(path,encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

"""
Shannon Entropy calulation
"""
def calcEntropy(x):
    p, lens = Counter(x), np.float(len(x))
    return -np.sum( count/lens * np.log2(count/lens) for count in p.values())

df['entropy'] = [calcEntropy(x) for x in df['lld']]

"""
LLD record lenght
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
Properties of the dataset
"""
#print('%d %d' % (df.shape[0], df.shape[1]))
#print('Total domains %d' % df.shape[0])

"""
Pearson Spearman correlation
Is there a correlation/linear correlation between domain name length and entropy?
"""
sns.set_context(rc={"figure.figsize": (7, 5)})
g = sns.JointGrid(df.length.astype(float), df.entropy.astype(float))
g.plot(sns.regplot, sns.distplot, stats.spearmanr);
print("Pearson's r: {0}".format(stats.pearsonr(df.length.astype(float), df.entropy.astype(float))))
plt.show()


"""
Nominal parametric upper
"""
#Regular DNS
dfNominal = df[df['label']== 0]
##DNS exfill
dfDGA = df[df['label']== 1]

def shadedHist(df,col,bins):
    df[col].hist(bins = bins, color = 'dodgerblue', alpha = .6, normed = False)
    len_mean = df[col].mean()
    len_std = df[col].std()

    # mpl red is 3 standard deviations
    plt.plot([len_mean, len_mean], [0,2500 ],'k-',lw=3,color = 'black',alpha = .4)
    plt.plot([len_mean + (2 * len_std), len_mean + (2 * len_std)], [0, 2500], 'k-', lw=2, color = 'red', alpha = .4)
    plt.axvspan(len_mean + (2 * len_std), max(df[col]), facecolor='r', alpha=0.3)
    plt.title(col)

"""
Nominal entropy distribution
"""
sns.set_context(rc={"figure.figsize": (7, 5)})

shadedHist(df[df['label']== 0],'entropy',3)
plt.show()


nominal_parametric_upper = dfNominal['entropy'].mean() + \
      2 * dfNominal['entropy'].std()

print("upper",nominal_parametric_upper)

if not dfDGA.empty:
    """
    Malicious entropy distribution
    """
    sns.set_context(rc={"figure.figsize": (7, 5)})
    shadedHist(dfDGA,'entropy',3)
    plt.show()


"""
Simple Decison Tree
"""

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
x = df.drop(['label','lld'], axis=1)
y = df['label']

# Training set size is 20%
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)

print("Accuracy score: ",dt.score(x_test,y_test))


"""
Performance
- Confusion matrix
- Classification report
- ROC
"""
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score

y_pred = dt.predict(x_test)
y_true = y_test

#Confusion matrix
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

#Classifucation report
print(classification_report(y_test, y_pred))

y_pred_proba = classifier.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Decision Tree ROC curve')
plt.show()
print('Area under the ROC Curve %d' % roc_auc_score(y_test,y_pred_proba))


"""
Cross validation k-fold
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=100)
model_kfold = DecisionTreeClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))

"""
Visualisation Decision tree
"""

if args.print:
    dot_data = StringIO()
    export_graphviz(classifier, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('tree.png')
