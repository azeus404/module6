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
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='domainlist')
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
print('Total llds %d' % data_total[0])

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

if args.deploy:
    print("[+] Model ready for deployment")
    joblib.dump(svm, 'models/svm_model.pkl')


"""
Performance

"""
y_pred = svm.predict(x_test)
y_true = y_test

print("[+]Confusion matrix")
print(pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))

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
plt.savefig('img/cm_svm.png')
plt.show()

print("[+]classification report")
target_names = ['Malicious', 'Benign']
report = classification_report(y_test, y_pred,target_names=target_names,output_dict=True)
print(pd.DataFrame(report).transpose())
print("True positive rate = Recall")


print("[+] ROC")
y_pred_proba = svm.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.plot(fpr,tpr, label='Support Vector Machine')
plt.legend()
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.title('Support Vector machine ROC curve')
plt.savefig('img/roc_svm.png')
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

"""
    Validation Curve
    https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/
"""

print("[+] Validation Curve")
x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

param_range = np.logspace(-6, -1, 5)
train_scores, test_scores = validation_curve(
    SVC(), x, y, param_name="gamma", param_range=param_range,
    scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.title("Validation Curve with SVM")
plt.xlabel(r"$\gamma$")
plt.ylabel("Score")
plt.ylim(0.0, 1.1)
lw = 2
plt.semilogx(param_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)
plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)
plt.legend(loc="best")
plt.savefig('img/validation_curve_svm.png')
plt.show()
