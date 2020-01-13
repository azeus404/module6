import pandas as pd
import pickle
import argparse
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib


parser = argparse.ArgumentParser(description='Process a lld')
parser.add_argument('lld', help='input lld')
parser.add_argument('--model', help='select trained model')
args = parser.parse_args()
lld = args.lld
model = args.model

print("Implement trained model")

dt_model = open('dt_model.pkl','rb')
dtc = joblib.load(dt_model)
dfa = dict()

"""
Shannon Entropy calulation
"""
def calcEntropy(x):
    p, lens = Counter(x), np.float(len(x))
    return -sum( count/lens * np.log2(count/lens) for count in p.values())

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

df = pd.DataFrame(data=dfa,index=[0])
prediction = dtc.predict(df)
if prediction ==[0]:
    print("This is benign: %s LLD" % lld)
else:
    print("This is posible malicious: %s LLD" % lld)
