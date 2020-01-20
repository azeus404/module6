import pandas as pd
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from collections import Counter
from warnings import filterwarnings
filterwarnings('ignore')
from tabulate import tabulate

import argparse
import joblib

from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='path to file with features added to domainlist')
parser.add_argument('--deploy', help='export model for deployment')
args = parser.parse_args()
path = args.path
deploy = args.deploy


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
print('Total llds %d' % data_total[0])

"""
Random forest
"""
print("[+] Applying Random Forest")
x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

#create a test set of size of about 20% of the dataset
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42 ,stratify=y)

rf=RandomForestClassifier(n_estimators=10,random_state=1)
rf.fit(x_train,y_train)

print("Accuracy score: %.2f" % rf.score(x_test,y_test))

if args.deploy:
    print("[+]Model ready for deployment")
    joblib.dump(rf, 'models/rf_model.pkl')

"""
Feature importance
"""

# Sort feature importances in descending order
# # Rearrange feature names so they match the sorted feature importances
print("[+] Feature importance")
headers = ["name", "score"]
values = sorted(zip(df.columns, rf.feature_importances_), key=lambda x: x[1] * -1)
print(tabulate(values, headers, tablefmt="plain"))

"""
Performance
- Confusion matrix
- Classification report
- ROC
- Precision recall curve
"""
y_pred = rf.predict(x_test)
y_true = y_test

print("[+] Confusion matrix")
print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

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
ax.set_title('Confusion Matrix - RF')
ax.xaxis.set_ticklabels(['negative', 'positive'])
ax.yaxis.set_ticklabels(['negative', 'positive'])
plt.savefig('img/cm_rf.png')
plt.show()

print("[+] classification report")
target_names = ['Malicious', 'Benign']
report = classification_report(y_test, y_pred,target_names=target_names,output_dict=True)
print(pd.DataFrame(report).transpose())
print("True positive rate = Recall")

print("[+] ROC")
y_pred_proba = rf.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.plot(fpr,tpr, label='Random Forest Classifier')
plt.legend()
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.title('Random Forest ROC curve')
plt.savefig('img/roc_rf.png')
plt.show()

#http://gim.unmc.edu/dxtests/ROC3.htm
print('Area under the ROC Curve %.2f' % roc_auc_score(y_test,y_pred_proba))
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")


print("[+] Precision recall curve --imbalanced dataset")
# precision-recall curve and f1 for an imbalanced dataset
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score,auc

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=42)
# fit a model
model = RandomForestClassifier()
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(testX)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(testX)
# calculate precision and recall for each threshold
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
# calculate scores
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('RandomForestClassifier: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='RandomForestClassifier')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('img/prc_rf.png')
plt.show()
"""
Cross validation k-fold
https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85
"""
print("[+] Cross validation")
kfold = KFold(n_splits=10, random_state=42)
model_kfold = RandomForestClassifier()
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
