import pandas as pd
from pandas.plotting import scatter_matrix
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter

import argparse


parser = argparse.ArgumentParser(description='Process labeled lld list')
parser.add_argument('path', help='location of lld list')
parser.add_argument('--out', help='export dataset with features')

args = parser.parse_args()
path = args.path
out = args.out


"""
    EXPLORATORY DATA ANALYSIS EDA -
    https://blog.floydhub.com/introduction-to-anomaly-detection-in-python/
"""

""""
Pre-process data: drop duplicates and empty
"""
df = pd.read_csv(path,encoding='utf-8')
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
Number of unique character in subdomain

df['numunique'] = [len(set(x)) for x in df['lld']]
"""

"""
Metric and statistics of the dataset
"""
data_total = df.shape
print('%d %d' % (data_total[0], data_total[1]))

print(df.describe().transpose())

# % of the dataset is malicious
print('percent of the dataset is malicious %d' % ((len(df.loc[df.label==1])) / (len(df.loc[df.label == 0])) * 100))

print('[*] Entropy values of total dataset')
print('Minimum length ' + str(df['length'].min()))
print('Maximum length  ' + str(df['length'].max()))

print('Minimum Entropy ' + str(df['entropy'].min()))
print('Maximum Entropy ' + str(df['entropy'].max()))


"""
    create the correlation matrix heat map
"""

plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(),linewidths=.1,cmap="YlGnBu", annot=True)
plt.yticks(rotation=0);
plt.title("Correlation matrix (Heat map)")
plt.show()


"""
Count by label
"""

plt.figure(figsize=(7, 5))
plt.title("Distribution of DNS domains Malicious vs Benign")
sns.countplot(df.label)
plt.show()

plt.figure(figsize=(11,11))
plt.title("Distribution of DNS domains Malicious vs Benign")
labels=df["label"].value_counts().index
sizes=df["label"].value_counts().values
plt.pie(sizes,labels=("Benign","Malicious"),autopct="%1.f%%")
plt.title("Value counts of label",size=25)
plt.legend()
plt.show()
print("Numbers of Value counts\n",df.loc[:,'label'].value_counts())

"""
Count by label
"""
#plt.axis('equal');
#plt.pie(sums, labels=sums.index);
#plt.show()
#plt.pie(sizes, labels = labels, autopct = "%.2f")
#plt.axes().set_aspect("equal")
#plt.show()

"""
Pearson Spearman correlation
Is there a correlation/linear correlation between domain name length and entropy?
https://wiki.uva.nl/methodologiewinkel/index.php/Spearman_correlation

"""
sns.set_context(rc={"figure.figsize": (7, 5)})
g = sns.JointGrid(df.length.astype(float), df.entropy.astype(float))
g.plot(sns.regplot, sns.distplot, stats.spearmanr);
print("Pearson's r: {0}".format(stats.pearsonr(df.length.astype(float), df.entropy.astype(float))))
plt.show()


"""
Benign parametric upper
"""

#Regular DNS
dfBenign = df[df['label']== 0]

##DNS exfill

dfMalicious= df[df['label']== 1]

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
Benign entropy distribution
"""

sns.set_context(rc={"figure.figsize": (7, 5)})
shadedHist(df[df['label']== 0],'entropy',3)
plt.show()


nominal_parametric_upper = dfBenign['entropy'].mean() + \
      2 * dfBenign['entropy'].std()

print("Benign upper entropy",nominal_parametric_upper)

if not dfMalicious.empty:
    """
    Malicious entropy distribution
    """
    sns.set_context(rc={"figure.figsize": (7, 5)})
    shadedHist(dfMalicious,'entropy',3)
    plt.show()

"""

"""
sns.pairplot(df)
plt.show()

"""
 Box plots Benign dataset
"""

sns.set(style="whitegrid")
length = dfBenign['length']
entropy = dfBenign['entropy']
numbchars = dfBenign['numbchars']
numbdots = dfBenign['numbdots']
numunique = dfBenign['numunique']

plt.title("Benign dataset")
sns.boxplot(x=length)
plt.show()

plt.title("Benign dataset")
sns.boxplot(x=entropy)
plt.show()

plt.title("Benign dataset")
sns.boxplot(x=numbchars)
plt.show()

plt.title("Benign dataset")
sns.boxplot(x=numbdots)
plt.show()

plt.title("Benign dataset")
sns.boxplot(x=numunique)
plt.show()

"""
  Box plots Malicious
"""

sns.set(style="whitegrid")
length = dfMalicious['length']
entropy = dfMalicious['entropy']
numbchars = dfMalicious['numbchars']
numbdots = dfMalicious['numbdots']
numunique = dfMalicious['numunique']

plt.title("Malicious dataset")
sns.boxplot(x=length)
plt.show()

plt.title("Malicious dataset")
sns.boxplot(x=entropy)
plt.show()

plt.title("Malicious dataset")
sns.boxplot(x=numbchars)
plt.show()

plt.title("Malicious dataset")
sns.boxplot(x=numbdots)
plt.show()

plt.title("Malicious dataset")
sns.boxplot(x=numunique)
plt.show()

"""
 Group by Class
"""
print(df.groupby('label').size())


"""
  Histograms  Benign
"""
plt.title("Benign")
sns.distplot(dfBenign['length']);
plt.show()

plt.title("Benign")
sns.distplot(dfBenign['entropy']);
plt.show()

plt.title("Benign")
sns.distplot(dfBenign['numbchars']);
plt.show()

plt.title("Benign")
sns.distplot(dfBenign['numbdots']);
plt.show()

plt.title("Benign")
sns.distplot(dfBenign['numunique']);
plt.show()

"""
    Histograms Malicious
"""
plt.title("Malicious")
sns.distplot(dfMalicious['length']);
plt.show()

plt.title("Malicious")
sns.distplot(dfMalicious['entropy']);
plt.show()

plt.title("Malicious")
sns.distplot(dfMalicious['numbchars']);
plt.show()

plt.title("Malicious")
sns.distplot(dfMalicious['numbdots']);
plt.show()

plt.title("Malicious")
sns.distplot(dfMalicious['numunique']);
plt.show()

"""
Export dataset to csv
"""
if args.out:
    # Export to csv
    df.to_csv(out,index=False)
