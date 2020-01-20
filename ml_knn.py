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
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

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
print("[+] Properties of the dataset")
data_total = df.shape
print('Total llds %d' % df.shape[0])



"""
KNN
"""

print("[+] Applying K-NN")
x = df.drop(['label','lld'],axis=1).values
y = df['label'].values


#preprocessing
normalized_x = preprocessing.normalize(x)

#create a test set of size of about 20% of the dataset

x_train,x_test,y_train,y_test = train_test_split(normalized_x ,y,test_size=0.2,random_state=42, stratify=y,shuffle=True)

knn = KNeighborsClassifier(n_jobs=-1)
print(knn.get_params())
#Setup a knn classifier with k neighbors
knn = KNeighborsClassifier()

#train
knn.fit(x_train,y_train)
knn.score(x_test,y_test)
print("Accuracy score: %.2f" % knn.score(x_test,y_test))

print("[+] Applying KNN tuning")
from sklearn.model_selection import GridSearchCV

param_grid = {'n_neighbors':[5,6,7,8,9,10],
          'leaf_size':[1,2,3,5],
          'weights':['uniform', 'distance'],
          'algorithm':['auto', 'ball_tree','kd_tree','brute'],
          'n_jobs':[-1]}

grid = GridSearchCV(KNeighborsClassifier(), param_grid, refit = True, verbose = 3)

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

if args.deploy:
    print("[+] Model ready for deployment")
    joblib.dump(knn, 'models/knn_model.pkl')



"""
Performance
- Confusion matrix
- Classification report
- ROC
- precision recall curve
"""

y_pred = knn.predict(x_test)

print("[+]Confusion matrix")
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
ax.set_title('Confusion Matrix - KNN')
ax.xaxis.set_ticklabels(['negative', 'positive'])
ax.yaxis.set_ticklabels(['negative', 'positive'])
plt.savefig('img/cm_knn.png')
plt.show()

print("[+]classification report")
target_names = ['Malicious', 'Benign']
report = classification_report(y_test, y_pred,target_names=target_names,output_dict=True)
print(pd.DataFrame(report).transpose())
print("True positive rate = Recall")

print("[+] ROC")
y_pred_proba = knn.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)



plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.plot(fpr,tpr, label='K-Neighbors Classifier')
plt.legend()
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.title('k-NN(n_neighbors=3) ROC curve')
plt.savefig('img/roc_knn.png')
plt.show()

#http://gim.unmc.edu/dxtests/ROC3.htm
print('Area under the ROC Curve %.2f' % roc_auc_score(y_test,y_pred_proba))
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")

print("[+] Precision recall curve --imbalanced dataset")
# precision-recall curve and f1 for an imbalanced dataset
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score,auc

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(normalized_x , y, test_size=0.2, random_state=42)
# fit a model
model = KNeighborsClassifier()
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
print('KNeighbors: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='KNeighbors')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('img/prc_knn.png')
plt.show()


"""
Cross validation
"""
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors':np.arange(1,50)}
knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(x,y)

print("Best score %.2f" % knn_cv.best_score_)
print(knn_cv.best_params_)



"""
Cross validation k-fold
"""
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, random_state=42)
model_kfold = KNeighborsClassifier()
results_kfold = cross_val_score(model_kfold, normalized_x , y, cv=kfold)

print("Accuracy: %.2f%%" % (results_kfold.mean()*100.0))
