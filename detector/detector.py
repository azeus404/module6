import pandas as pd
import pickle
import argparse
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score,accuracy_score
from sklearn.tree import DecisionTreeClassifier
import joblib


parser = argparse.ArgumentParser(description='Process a lld and determin if its benign')
parser.add_argument('lld', help='input lld strinf')
parser.add_argument('--model', help='select a trained model')
parser.add_argument('--features',  type=int, help='select number of freatures')
args = parser.parse_args()
lld = args.lld
model = args.model


if args.model == 'dt':
    ml_model = open('../models/dt_model.pkl','rb')
elif args.model == 'knn':
    dt_model = open('../models/knn_model.pkl','rb')
elif args.model == 'logreg':
    ml_model = open('../models/logreg_model.pkl','rb')
elif args.model == 'nb':
    ml_model = open('../models/nb_model.pkl','rb')
elif args.model == 'nn':
    ml_model = open('../models/nn_model.pkl','rb')
elif args.model == 'rf':
    ml_model = open('../models/rf_model.pkl','rb')
elif args.model == 'svm':
    ml_model = open('../models/svm_model.pkl','rb')

print("[*] Implementing trained model: %s" % args.model)
dtc = joblib.load(ml_model)

dfa = dict()

if args.features == 2:
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
    dfa['numbchars'] = [countChar(x) for x in df['lld']]

elif args.features == 4:
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
    Number of . in subdomain
    """
    dfa['numbdots'] = [x.count('.') for x in lld]


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
    dfa['numbchars'] = [countChar(x) for x in lld]
else:
    print("[*] no features selected")


df = pd.DataFrame(data=dfa,index=[0])
print(df.head)
prediction = dtc.predict(df)
if prediction ==[0]:
    print("This is benign: %s LLD" % lld)
else:
    print("This is possible malicious: %s LLD" % lld)
