import pandas as pd
import numpy as np
import argparse

from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

"""

https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
https://machinelearningmastery.com/an-introduction-to-feature-selection/
https://machinelearningmastery.com/an-introduction-to-feature-selection/
There are five methods used to identify features to remove:
- Missing Values
- Single Unique Values
- Collinear Features
- Zero Importance Features
- Low Importance Features

In this script:
- Univariate Selection (ANOVA).
- Recursive Feature Elimination.
- Principle Component Analysis.


Three benefits of performing feature selection before modeling your data are:

1.Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
2.Improves Accuracy: Less misleading data means modeling accuracy improves.
3, Reduces Training Time: Less data means that algorithms train faster.

https://www.datacamp.com/community/tutorials/feature-selection-python

Added Recursive Feature Elimination with Cross-Validation (RFECV)
https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15

"""
print("[*] Loading data")
# load data
parser = argparse.ArgumentParser(description='Process features')
parser.add_argument('path', help='lld with features added')


args = parser.parse_args()
path = args.path
prefix = path.split('_')[3]
#df = pd.read_csv('../TRAININGS_DATA/lld_lab_tunnel_features_added.csv',encoding='utf-8')

df = pd.read_csv(path,encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

print("[*] Adding features Feature")
"""
Number of . in subdomain
"""
df['numbdots'] = [x.count('.') for x in df['lld']]

"""
Number of unique character in subdomain
"""
df['numunique'] = [len(set(x)) for x in df['lld']]

print("[*] Creating traing sets")

XL = df.drop(['label','lld'],axis=1)

X = df.drop(['label','lld'],axis=1).values
Y = df['label'].values

print("[*] Feature Selection with Univariate Statistical Tests (ANOVA)")
# Feature Selection with Univariate Statistical Tests
# feature extraction
test = SelectKBest(score_func=f_classif, k=3)
fvalue_selector = SelectKBest(f_classif, k=3)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
#print(fit.scores_)
features = fit.transform(X)

X_kbest = fvalue_selector.fit_transform(X, Y)
print('Original number of features:', X.shape[1])
print('Reduced number of features:', X_kbest.shape[1])

# Capture P values in a series
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=101)
univariate = f_classif(x_train, y_train)
univariate = pd.Series(univariate[1])
univariate.index = XL.columns
univariate.sort_values(ascending=False, inplace=True)
# Plot the P values
plt.title('Feature importance - ANOVA')
univariate.sort_values(ascending=False).plot.bar(figsize=(20,8))
plt.savefig('../img/'+prefix+'_feature_anova.png')
plt.show()

# summarize selected features
print(features[0:3,:])

print("[*] Feature Extraction with RFE")
# Feature Extraction with RFE

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print('Original number of features:', X.shape[1])
print("Optimized Number Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


print("[*] Feature Extraction with RFECV")
# Feature Extraction with RFECV

# feature extraction
rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, Y)

print('Original number of features:', X.shape[1])
print("Optimized Number Features: %d" % rfecv.n_features_)
print("Selected Features: %s" % rfecv.support_)
print("Feature Ranking: %s" % rfecv.ranking_)


plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)
plt.savefig('../img/'+prefix+'_feature_frecv.png')
plt.show()

print(np.where(rfecv.support_ == False)[0])
print("[*] Feature Extraction with PCA")
# Feature Extraction with PCA
pca = PCA(n_components=5)
fit = pca.fit(X)
# summarize components
# print('Original number of features:', X.shape[1])
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)



print("[*] Feature Importance with Extra Trees Classifier")
# Feature Importance with Extra Trees Classifier
# feature extraction

model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print('[+] Feature importance')
print('Original number of features:', X.shape[1])
print("Number Features: %d" % model.n_features_)

#plot graph of feature importances for better visualization
plt.figure(figsize=(16, 14))
plt.title('Feature importance - Extra Trees Classifier')
feat_importances = pd.Series(model.feature_importances_, index=df.drop(['label','lld'],axis=1).columns.values.tolist())
feat_importances.nlargest(10).plot(kind='barh')
plt.savefig('../img/'+prefix+'_feature_etc.png')
plt.show()

print('[+] Feature Importance Treesbased Classifier')
model_tree = RandomForestClassifier(random_state=100, n_estimators=50)
model_tree.fit(X, Y)
print(XL.columns.values)
print(model_tree.feature_importances_)

dset = pd.DataFrame()
dset['attr'] = XL.columns
dset['importance'] = model_tree.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 10))
plt.title('Feature importance - Treesbased Classifier')
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('Treesbased Classifier - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.savefig('../img/'+prefix+'_feature_tbc.png')
plt.show()
