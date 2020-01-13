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

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve

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
Support Vector machine

"""
print("[+] Applying Support Vector Machine")

svm = SVC(random_state = 1,gamma='auto',probability=True )

x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

#create a test set of size of about 20% of the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

svm.fit(x_train,y_train)

print("Accuracy score: ",svm.score(x_test,y_test))

"""
Performance

"""


y_pred = svm.predict(x_test)
y_true = y_test

print("[+]Confusion matrix")
print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))


print("[+]classification report")
#https://muthu.co/understanding-the-classification-report-in-sklearn/
print(classification_report(y_true, y_pred))


#ROC
y_pred_proba = svm.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.plot(fpr,tpr, label='Support Vector Machine')
plt.legend()
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.title('Support Vector machine ROC curve')
plt.show()
print('Area under the ROC Curve %d' % float(roc_auc_score(y_test,y_pred_proba)))
#http://gim.unmc.edu/dxtests/ROC3.htm
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")

"""
Cross validation k-fold
"""
print("[+] Cross validation")

kfold = KFold(n_splits=10, random_state=42)
model_kfold = SVC()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
