
import argparse

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
"""
    Bag of Words
"""

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')

args = parser.parse_args()
path = args.path

df = pd.read_csv(path,encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)







corpora = df['lld']
cvec = CountVectorizer(lowercase=False, ngram_range=(1,2))
wm = cvec.fit_transform(corpora)
tokens = cvec.get_feature_names()
print(tokens)
#print(wm2df(wm, tokens))
