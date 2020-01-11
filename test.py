import pandas as pd
import argparse

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold



""""
Pre-process data: drop duplicates and empty
"""

df = pd.read_csv('lld_lab_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print(df.head())
dt = DecisionTreeClassifier()

#features, labels
x = df.drop(['label','lld'], axis=1)
y = df['label']

# Training set size is 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, shuffle=True,  stratify=y)
dt.fit(x_train,y_train)

print("Accuracy score: ",dt.score(x_test,y_test))


"""
Cross validation k-fold
"""
print("[+] Cross validation")
kfold = KFold(n_splits=10, random_state=100)
model_kfold = DecisionTreeClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
