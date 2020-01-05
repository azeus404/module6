import pandas as pd
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
from warnings import filterwarnings
filterwarnings('ignore')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier


""""
Pre-process data: drop duplicates
"""
df = pd.read_csv('lld_labeled.csv',encoding='utf-8')
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
#Pearson#
"""
sns.set_context(rc={"figure.figsize": (7, 5)})
g = sns.JointGrid(df.length.astype(float), df.entropy.astype(float))
g.plot(sns.regplot, sns.distplot, stats.spearmanr);
print("Pearson's r: {0}".format(stats.pearsonr(df.length.astype(float), df.entropy.astype(float))))
plt.show()


"""
nominal_parametric_upper
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

"""
Malicious entropy distribution
"""
sns.set_context(rc={"figure.figsize": (7, 5)})
shadedHist(dfDGA,'entropy',3)
plt.show()


"""
N-gram
"""


cv = CountVectorizer(analyzer='char', ngram_range=(3,4))
cv_nominal = cv.fit_transform(df[df['label']== 0]['lld'])
cv_all     = cv.fit_transform(df['lld'])

feature_names = cv.get_feature_names()

#sorted(cv.vocabulary_.iteritems(), key=operator.itemgetter(1), reverse= True)[0:5]
dfConcat = pd.concat([df.ix[:, 2:4], pd.DataFrame(cv_all.toarray())], join='outer', axis=1, ignore_index=False)
#print(dfConcat.head(3))

"""
Cross_validation
"""
X = dfConcat.values
y = df.ix[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=22)


"""
DummyClassifier
"""

for strategy in ['stratified', 'most_frequent', 'uniform']:
    clf = DummyClassifier(strategy=strategy,random_state=None)
    clf.fit(X_train, y_train)
    print(strategy + ': ' + str(clf.score(X_test, y_test)))

"""
Random forrest 1st pass
"""

rf = RandomForestClassifier(n_jobs = -1)
rf.fit(X_train, y_train)
print('RF' + ': ' + str(rf.score(X_test, y_test)))

"""
English dictionary
"""
dfEng = pd.read_csv('eng_words.txt', names=['word'],
                    header=None, dtype={'word': np.str},
                    encoding='utf-8')
#Convert to lowercase
dfEng['word'] = dfEng['word'].map(lambda x: np.str(x).strip().lower())
dfEng['word'].drop_duplicates(inplace=True)
dfEng['word'].dropna(inplace=True)

"""
N-gram dictionary match
"""
cvEng = CountVectorizer(analyzer='char', ngram_range=(3,4))
cvEngfeatures = cvEng.fit_transform(dfEng['word'])

#sorted(cvEng.vocabulary_.iteritems(), key=operator.itemgetter(1), reverse= True)[0:5]

df['dictMatch'] = np.log10(cvEngfeatures.sum(axis=0).getA1()) * cvEng.transform(df['lld']).T
print(df.head(20))

"""
Random forrest 2nd pass
"""
dfConcat2 = pd.concat([pd.DataFrame(df.ix[:,4]),dfConcat],join='outer', axis=1, ignore_index=False)
X = dfConcat2.values
y = df.ix[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=None)

rf = RandomForestClassifier(n_jobs = -1)
rf.fit(X_train, y_train)
print('RF' + ': ' + str(rf.score(X_test, y_test)))
