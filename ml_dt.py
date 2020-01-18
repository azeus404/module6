import pandas as pd
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
from warnings import filterwarnings
filterwarnings('ignore')

import pydotplus
import argparse
import joblib

from sklearn.model_selection import train_test_split,cross_val_score,validation_curve
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score,accuracy_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', default='lld_lab_dnscat_4features_added.csv',help='domainlist')
parser.add_argument('--deploy', help='export model for deployment')
args = parser.parse_args()
path = args.path
deploy = args.deploy

""""
Pre-process data: drop duplicates and empty
"""
df = pd.read_csv(path,encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

"""
Properties of the dataset
"""
print("[+] Properties of the dataset")
print('Total llds %d' % df.shape[0])

"""
Simple Decison Tree
"""
print("[+] Applying Simple Decison Tree")
dt=DecisionTreeClassifier()

#features, labels
x = df.drop(['label','lld'], axis=1)
y = df['label']

m,n = x.shape[0],y.shape[1]
pos,neg= (y==1).reshape(m,1), (y==0).reshape(m,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+",s=50)
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],c="y",marker="o",s=50)
plt.show()
# Training set size is 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y,shuffle=True)

#train
dt.fit(x_train,y_train)
print("Accuracy score: ", dt.score(x_test,y_test))

if args.deploy:
    print("[+] Model ready for deployment")
    joblib.dump(dt, 'models/dt_model.pkl')

"""
Performance
- Confusion matrix
- Classification report
- ROC
"""
#evaluating test sets
y_pred = dt.predict(x_test)
y_true = y_test

print("[+]Confusion matrix")
#confusion_matrix(df.actual_label.values, df.predicted_RF.values)
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True))


#Confusion matrix to file
cf_matrix = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cf_matrix, range(2),
                  range(2))

# plot (powered by seaborn)
ax= plt.subplot()
sns.set(font_scale=1)
sns.heatmap(df_cm, ax = ax, annot=True,annot_kws={"size": 16}, fmt='g')

# labels, title and ticks
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['negative', 'positive'])
ax.yaxis.set_ticklabels(['negative', 'positive'])
plt.savefig('img/cm_dt.png')
plt.show()

print("[+]classification report")
target_names = ['Malicious', 'Benign']
report = classification_report(y_test, y_pred,target_names=target_names,output_dict=True)
print(pd.DataFrame(report).transpose())
print("True positive rate = Recall")

y_pred_proba = dt.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



print('Accuracy %d' % accuracy_score(y_test, y_pred ))

print('What percent of positive predictions were correct? F-1 %d' % f1_score(y_test, y_pred ))

print('Recall (TRP) %d' % recall_score(y_test, y_pred))

#https://towardsdatascience.com/understanding-data-science-classification-metrics-in-scikit-learn-in-python-3bc336865019
print("[+] ROC")
plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.plot(fpr,tpr, label='Decision Tree Classifier')
plt.legend()
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.title('Decision Tree ROC curve')
plt.savefig('img/roc_dt.png')
plt.show()


#http://gim.unmc.edu/dxtests/ROC3.htm
print('Area under the ROC Curve %d' % roc_auc_score(y_test,y_pred_proba))
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")

"""
Cross validation k-fold
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=42)
model_kfold = DecisionTreeClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
