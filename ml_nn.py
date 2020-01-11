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
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold

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
Neural network
"""
print("[+]Applying Neural Network")

mlp = MLPClassifier(hidden_layer_sizes=(8,8,8), activation='relu', solver='adam', max_iter=500)


x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

#create a test set of size of about 20% of the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)
mlp.fit(x_train,y_train)
predict_train = mlp.predict(x_train)
predict_test = mlp.predict(x_test)

print("Accuracy score: ",mlp.score(x_test,y_test))

"""
Performance
- Confusion matrix
- Classification report
- ROC
"""

y_pred = mlp.predict(x_test)
y_true = y_test

print("[+]Confusion matrix")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

print("[+]classification report")
print(classification_report(y_test, y_pred))

#ROC
y_pred_proba = mlp.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Neural Network ROC curve')
plt.show()
print('Area under the ROC Curve %d' % roc_auc_score(y_test,y_pred_proba))
#http://gim.unmc.edu/dxtests/ROC3.htm
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")


"""
Cross validation k-fold
"""
kfold = KFold(n_splits=10, random_state=42)
model_kfold = MLPClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))