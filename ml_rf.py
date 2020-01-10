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
from sklearn.ensemble import RandomForestClassifier


parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')

args = parser.parse_args()
path = args.path



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
Pearson Spearman correlation
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
Random forrest
"""


x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

rt=RandomForestClassifier(n_estimators=35,random_state=1)
rt.fit(x_train,y_train)

print("score: ",rt.score(x_test,y_test))

score_list2=[]
for i in range(1,50):
    rt2=RandomForestClassifier(n_estimators=i,random_state=1)
    rt2.fit(x_train,y_train)
    score_list2.append(rt2.score(x_test,y_test))

plt.figure(figsize=(12,8))
plt.plot(range(1,50),score_list2)
plt.xlabel("Esimator values")
plt.ylabel("Acuracy")
plt.show()

"""
Performance

"""
from sklearn.metrics import classification_report, confusion_matrix

y_pred = rt.predict(x_test)
y_true = y_test

#Confusion matrix
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

#Classifucation report
print(classification_report(y_test, y_pred))

"""
Cross validation k-fold
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=100)
model_kfold = RandomForestClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
