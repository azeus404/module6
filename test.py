import pandas as pd
import pickle
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals import joblib

df = pd.read_csv('lld_lab_dnscat_features_added.csv',encoding='utf-8')
dfa = pd.DataFrame(columns=['entropy','length','numbchars','numbdots'])

"Trainingsdata"
#https://thispointer.com/python-pandas-how-to-add-rows-in-a-dataframe-using-dataframe-append-loc-iloc/
df = pd.read_csv('lld_lab_dnscat_features_added.csv',encoding='utf-8')



# Features and Labels
x = df.drop(['label','lld'], axis=1)
y = df['label']

#train 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#DT Classifier
dt=DecisionTreeClassifier()

# Extract
X = dt.fit(x_train,y_train)
dt.score(x_test,y_test)

lld = '3e212139008dadada'
"""
Shannon Entropy calulation
"""
def calcEntropy(x):
    p, lens = Counter(x), np.float(len(x))
    return -np.sum( count/lens * np.log2(count/lens) for count in p.values())

dfa['entropy'] = calcEntropy(lld)

"""
LLD record length
"""
dfa['length'] = len(lld)


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
dfa['numbchars'] = countChar(lld)

"""
Number of . in subdomain
"""
dfa['numbdots'] = lld.count('.')
print(dfa.head)
#print(prediction = dt.predict(dfa.values))
