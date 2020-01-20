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

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import validation_curve
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score, roc_curve,recall_score

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='path to file with features added to domainlist')
parser.add_argument('--deploy', help='export model for deployment')
args = parser.parse_args()
path = args.path
deploy = args.deploy

"""
    tuning https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/
"""


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

print(svm.get_params())

x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

#preprocessing
normalized_x = preprocessing.normalize(x)

#create a test set of size of about 20% of the dataset
x_train,x_test,y_train,y_test = train_test_split(normalized_x,y,test_size=0.2,random_state=42, stratify=y)

svm.fit(x_train,y_train)

y_pred = svm.predict(x_test)
y_true = y_test

print('Recall (TRP) %.2f (1 = best 0 = worse)' % recall_score(y_test, y_pred))
print("Accuracy score: %.2f" % svm.score(x_test,y_test))


if args.deploy:
    print("[+] Model ready for deployment")
    joblib.dump(svm, 'models/svm_model.pkl')


print("[+] Applying Support Vector Machine tuning")
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['linear','rbf']}

grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)


grid_predictions = grid.predict(x_test)

# print classification report
print(classification_report(y_test, grid_predictions))

# Print the tuned parameters and score
print("Tuned SVM Parameters: {}".format(grid.best_params_))
print("Best score is {}".format(grid.best_score_))



"""
Performance
- Confusion matrix
- Classification report
- ROC
- Precision recall curve
"""


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
ax.set_title('Confusion Matrix - SVM')
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
print('Area under the ROC Curve %.2f' % roc_auc_score(y_test,y_pred_proba))
#http://gim.unmc.edu/dxtests/ROC3.htm
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")

print("[+] Precision recall curve --imbalanced dataset")
# precision-recall curve and f1 for an imbalanced dataset
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score,auc

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(normalized_x, y, test_size=0.2, random_state=42)
# fit a model
model = SVC(probability=True )
model.fit(trainX, trainy)
# predict probabilities
lr_probs = model.predict_proba(x_test)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]
# predict class values
yhat = model.predict(testX)
# calculate precision and recall for each threshold
lr_precision, lr_recall, _ = precision_recall_curve(testy, lr_probs)
# calculate scores
lr_f1, lr_auc = f1_score(testy, yhat), auc(lr_recall, lr_precision)
# summarize scores
print('SVC: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='SVC')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('img/prc_svm,.png')
plt.show()

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
