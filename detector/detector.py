import pandas as pd
import pickle
import argparse
import numpy as np
from collections import Counter
import joblib


parser = argparse.ArgumentParser(description='Process a lld and determin if its benign')
parser.add_argument('lld', help='input lld string')
parser.add_argument('--model',default='rf', help='select a trained model')
args = parser.parse_args()
lld = args.lld
model = args.model


if args.model == 'rf':
    ml_model = open('../models/rf_model.pkl','rb')

print("[*] Implementing trained model: %s" % args.model)

#load trained model
tm = joblib.load(ml_model)
dfa = dict()


print("[*] Extracting features: entropy,length and Ratio between alpha and numbers in string")
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
Ratio between alpha and numbers in string
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

print("[*] Making a prediction:")
prediction = tm.predict(df)

if prediction ==[0]:
    print("This is benign: %s LLD" % lld)
else:
    print("This is possible malicious: %s LLD" % lld)
