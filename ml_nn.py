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
from sklearn.metrics import classification_report, confusion_matrix,roc_curve,roc_auc_score,recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

parser = argparse.ArgumentParser(description='Process lld_labeled')
parser.add_argument('path', help='path to file with features added to domainlist')
parser.add_argument('--deploy', help='export model for deployment')
args = parser.parse_args()
path = args.path
deploy = args.deploy

f = open("scores/nn_scores.txt", "w")

"""
    Tuning https://www.geeksforgeeks.org/ml-hyperparameter-tuning/
"""

""""
Pre-process data: drop duplicates
"""
df = pd.read_csv(path,encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df = shuffle(df).reset_index(drop=True)
f.write('Dataset %s\n' % path)

"""
Properties of the dataset
"""
data_total = df.shape
print('Total llds %d' % data_total[0])
f.write('Total llds %d \n' % data_total[0])

"""
Neural network
"""
print("[+] Applying Neural Network")

model = MLPClassifier()

print(model.get_params())

x = df.drop(['label','lld'],axis=1).values
y = df['label'].values

#create a test set of size of about 20% of the dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

#Standardizing numeric variables.
scale = preprocessing.StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)

model.fit(x_train,y_train)

predict_train = model.predict(x_train)
predict_test = model.predict(x_test)

y_pred = model.predict(x_test)
y_true = y_test

print('Recall (TRP) %.2f (1 = best 0 = worse)' % recall_score(y_test, y_pred))
print("Untuned accuracy score: %.2f" % model.score(x_test,y_test))
f.write("Untuned accuracy score: %.2f \n" % model.score(x_test,y_test))


print("[+] Applying neural network tuning")
from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {
        'hidden_layer_sizes': [(7, 7), (128,), (128, 7)],
        'tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'epsilon': [1e-3, 1e-7, 1e-8, 1e-9, 1e-8]
    }

grid = GridSearchCV(model, param_grid, refit = True)

# fitting the model for grid search
grid.fit(x_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
f.write('tunned parameters %s \n' % str(grid.get_params()))

grid_predictions = grid.predict(x_test)

# print classification report
print(classification_report(y_test, grid_predictions))

# Print the tuned parameters and score
print("Tuned NN Parameters: {}".format(grid.best_params_))
print("Best score is {}".format(grid.best_score_))

f.write('Tuned parameters %s \n' % str(grid.get_params()))
f.write("Tuned accuracy score: %.2f \n" % (grid.best_score_*100.0))

if args.deploy:
    print("[+] Model ready for deployment")
    model = MLPClassifier(**grid.best_params_)
    model.fit(x_train,y_train)
    joblib.dump(model, 'models/nn_model.pkl')


"""
Performance
- Confusion matrix
- Classification report
- ROC
- Precision recall curve
"""

y_pred = model.predict(x_test)
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
ax.set_title('Confusion Matrix - NN')
ax.xaxis.set_ticklabels(['negative', 'positive'])
ax.yaxis.set_ticklabels(['negative', 'positive'])
plt.savefig('img/cm_nn.png')
plt.show()

print("[+]classification report")
target_names = ['Malicious', 'Benign']
report = classification_report(y_test, y_pred,target_names=target_names,output_dict=True)
print(pd.DataFrame(report).transpose())
print("True positive rate = Recall")

print("[+] ROC")
y_pred_proba = model.predict_proba(x_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


plt.plot([0,1],[0,1],'k-',label='random')
plt.plot([0,0,1,1],[0,1,1,1],'g-',label='perfect')
plt.plot(fpr,tpr, label='Neural Network')
plt.legend()
plt.xlabel('False Positive Rate - FPR')
plt.ylabel('True Positive Rate - TPR')
plt.title('Neural Network ROC curve')
plt.savefig('img/roc_nn.png')
plt.show()

print('Area under the ROC Curve %.2f' % roc_auc_score(y_test,y_pred_proba))
#http://gim.unmc.edu/dxtests/ROC3.htm
print(".90-1 = excellent (A) .80-.90 = good (B) .70-.80 = fair (C) .60-.70 = poor (D) .50-.60 = fail (F)")

print("[+] Precision recall curve --imbalanced dataset")
# precision-recall curve and f1 for an imbalanced dataset
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score,auc

# split into train/test sets
trainX, testX, trainy, testy = train_test_split(x, y, test_size=0.2, random_state=42)
# fit a model
model = MLPClassifier(**grid.best_params_)
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
print('MLPClassifier: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# plot the precision-recall curves
no_skill = len(testy[testy==1]) / len(testy)
plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
plt.plot(lr_recall, lr_precision, marker='.', label='MLPClassifier')
# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
plt.savefig('img/prc_nn.png')
plt.show()


"""
Cross validation k-fold
"""
kfold = KFold(n_splits=10, random_state=42)
model_kfold = MLPClassifier(**grid.best_params_)
results_kfold = cross_val_score(model_kfold, x, y, cv=kfold)

print("Cross validated accuracy: %.2f%%" % (results_kfold.mean()*100.0))
f.write("Cross validated accuracy: %.2f \n" % (results_kfold.mean()*100.0))
f.close()
