import pandas as pd
import operator
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings('ignore')

import argparse
import joblib

from sklearn import preprocessing
from sklearn.model_selection import train_test_split,cross_val_score,validation_curve
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score,accuracy_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='path to file with features added to domainlist')
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
df =  shuffle(df).reset_index(drop=True)

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

# preprocessing
standardized_x = preprocessing.scale(x)

# Training set size is 20%
x_train, x_test, y_train, y_test = train_test_split(standardized_x , y, test_size=0.20, random_state=42, stratify=y,shuffle=True)

#train
dt.fit(x_train,y_train)
print("Accuracy score: %.2f" % dt.score(x_test,y_test))

if args.deploy:
    print("[+] Model ready for deployment")
    joblib.dump(dt, 'models/dt_model.pkl')

"""
Performance
- Confusion matrix
- Classification report
- ROC
- Precision recall curve
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
ax.set_title('Confusion Matrix - DT')
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



print('Accuracy %.2f' % accuracy_score(y_test, y_pred ))

print('What percent of positive predictions were correct? F-1 %.2f' % f1_score(y_test, y_pred ))

print('Recall (TRP) %.2f' % recall_score(y_test, y_pred))

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
print('Area under the ROC Curve %.2f' % roc_auc_score(y_test,y_pred_proba))
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")


print("[+] Precision recall curve --imbalanced dataset")
# precision-recall curve and f1 for an imbalanced dataset
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score,auc

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(standardized_x, y, test_size=0.2, random_state=42,shuffle=True)
# fit a model
model = DecisionTreeClassifier()
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
print('Decision Tree Classifier: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='Decision Tree Classifier')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('img/prc_dt.png')
plt.show()


"""
Cross validation k-fold
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=42)
model_kfold = DecisionTreeClassifier()
results_kfold = cross_val_score(model_kfold, standardized_x, y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
