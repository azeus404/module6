import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

"""

https://towardsdatascience.com/a-feature-selection-tool-for-machine-learning-in-python-b64dd23710f0
https://machinelearningmastery.com/an-introduction-to-feature-selection/
There are five methods used to identify features to remove:
https://machinelearningmastery.com/an-introduction-to-feature-selection/

Missing Values
Single Unique Values
Collinear Features
Zero Importance Features
Low Importance Features

In this script:
Univariate Selection.
Recursive Feature Elimination.
Principle Component Analysis.
Feature Importance.

Three benefits of performing feature selection before modeling your data are:

Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.
Improves Accuracy: Less misleading data means modeling accuracy improves.
Reduces Training Time: Less data means that algorithms train faster.

https://www.datacamp.com/community/tutorials/feature-selection-python

Added Recursive Feature Elimination with Cross-Validation (RFECV)
https://towardsdatascience.com/feature-selection-in-python-recursive-feature-elimination-19f1c39b8d15

Lassoo
"""
print("[*] Loading data")
# load data
df = pd.read_csv('../TRAININGS_DATA/lld_lab_dnscat_features_added.csv',encoding='utf-8')
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

print("[*] Feature Selection with Univariate Statistical Tests")
# Feature Selection with Univariate Statistical Tests
# feature extraction
test = SelectKBest(score_func=f_classif, k=3)
fit = test.fit(X, Y)
# summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:3,:])

print("[*] Feature Extraction with RFE")
# Feature Extraction with RFE

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


print("[*] Feature Extraction with RFECV")
# Feature Extraction with RFE

# feature extraction
rfc = RandomForestClassifier(random_state=101)
rfecv = RFECV(estimator=rfc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(X, Y)

print("Number Features: %d" % rfecv.n_features_)
print("Selected Features: %s" % rfecv.support_)
print("Feature Ranking: %s" % rfecv.ranking_)

plt.figure(figsize=(16, 9))
plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
plt.ylabel('% Correct Classification', fontsize=14, labelpad=20)
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_, color='#303F9F', linewidth=3)

plt.show()
print(np.where(rfecv.support_ == False)[0])
print("[*] Feature Extraction with PCA")
# Feature Extraction with PCA

# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


print("[*] Feature Importance with Extra Trees Classifier")
# Feature Importance with Extra Trees Classifier
# feature extraction
print('[+] Feature importance')
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)

#plot graph of feature importances for better visualization

feat_importances = pd.Series(model.feature_importances_, index=df.drop(['label','lld'],axis=1).columns.values.tolist())
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

print('[+] Feature Importance Treesbased Classifier')
model_tree = RandomForestClassifier(random_state=100, n_estimators=50)
model_tree.fit(X, Y)
print(model_tree.feature_importances_)

dset = pd.DataFrame()
dset['attr'] = XL.columns
dset['importance'] = model_tree.feature_importances_

dset = dset.sort_values(by='importance', ascending=False)

plt.figure(figsize=(16, 14))
plt.barh(y=dset['attr'], width=dset['importance'], color='#1976D2')
plt.title('Treesbased Classifier - Feature Importances', fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Importance', fontsize=14, labelpad=20)
plt.show()
