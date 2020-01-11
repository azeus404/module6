import pandas as pd
from pandas.plotting import scatter_matrix
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter

import argparse
import tldextract

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')
parser.add_argument('--out', help='export dataset')

args = parser.parse_args()
path = args.path
out = args.out


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
Metric and statistics of the dataset
"""
data_total = df.shape
print('%d %d' % (data_total[0], data_total[1]))

print(df.describe().transpose())


print('Minimum length ' + str(df['length'].min()))
print('Maximum length  ' + str(df['length'].max()))

print('Minimum Entropy ' + str(df['entropy'].min()))
print('Maximum Entropy ' + str(df['entropy'].max()))

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
Box plots
"""

df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

"""
 Group by Class
"""
print(df.groupby('label').size())

"""
    histograms
"""
df.hist()
plt.show()


"""

"""

scatter_matrix(df)
plt.show()

"""
Entropy compared scatter plot
Below you can see that our DGA domains do tend to have higher entropy than benign domains on average.


malicious = df['label'] == 1
benign = df['label'] == 0
plt.scatter(benign['length'],benign['entropy'], s=140, c='#aaaaff', label='Benign', alpha=.2)
plt.scatter(malicious['length'], malicious['entropy'], s=40, c='r', label='Malicious', alpha=.3)
plt.legend()
pylab.xlabel('Domain Length')
pylab.ylabel('Domain Entropy')
plt.show()
"""
"""
Export dataset to csv
"""
if args.out:
    # Export to csv
    df.to_csv(out,index=False)