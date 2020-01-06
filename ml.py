import pandas as pd
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
from warnings import filterwarnings
filterwarnings('ignore')

import argparse

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')
parser.add_argument('out', help='lldlist')

args = parser.parse_args()
path = args.path
out = args.out


""""
Pre-process data: drop duplicates
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
df['length'] = [len(x) for x in df['lld']]



"""
#Pearson# Spearman correlation
Is there a correlation/linear correlation between domain name length and entropy?
"""
sns.set_context(rc={"figure.figsize": (7, 5)})
g = sns.JointGrid(df.length.astype(float), df.entropy.astype(float))
g.plot(sns.regplot, sns.distplot, stats.spearmanr);
print("Pearson's r: {0}".format(stats.pearsonr(df.length.astype(float), df.entropy.astype(float))))
plt.show()


"""
Nominal_parametric_upper
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
Simple decison tree
https://stackabuse.com/decision-trees-in-python-with-scikit-learn/
"""
from sklearn.model_selection import train_test_split

X = df.drop(['label','lld'], axis=1)
y = df['label']
print(df.head(10))
print(df.describe())

# Training set size is 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)


"""
Preformance

"""
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Export to csv
#df.to_csv(out,index=False)
