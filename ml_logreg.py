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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score

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
Properties of the dataset
"""
data_total = df.shape
print('%d %d' % (data_total[0], data_total[1]))
print('Total domains %d' % data_total[0])

"""
Logistic Regression
"""


x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

#create a test set of size of about 20% of the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42, stratify=y)

lr=LogisticRegression(solver='lbfgs')
lr.fit(x_train,y_train)

print("Accuracy score: ",lr.score(x_test,y_test))


"""
Performance
- Confusion matrix
- Classification report
- ROC
"""


y_pred = lr.predict(x_test)
y_true = y_test

print("[+]Confusion matrix")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

print("[+]classification report")
print(classification_report(y_test, y_pred))

#ROC
y_pred_proba = lr.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Logistic Regression ROC curve')
plt.show()
print('Area under the ROC Curve %d' % roc_auc_score(y_test,y_pred_proba))

"""
Cross validation k-fold
"""

kfold = KFold(n_splits=10, random_state=100)
model_kfold = LogisticRegression()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
