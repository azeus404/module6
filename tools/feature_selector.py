
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


Lassoo
"""

print("[*] Feature Selection with Univariate Statistical Tests")
# Feature Selection with Univariate Statistical Tests
import pandas as pd
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest,f_classif

import matplotlib.pyplot as plt

# load data
df = pd.read_csv('../lld_lab_dnscat_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
X = df.drop(['label','lld'],axis=1).values
Y = df['label'].values

# feature extraction
test = SelectKBest(score_func=f_classif, k=3)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)

# summarize selected features
print(features[0:3,:])

print("[*] Feature Extraction with RFE")
# Feature Extraction with RFE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# load data
df = pd.read_csv('../lld_lab_dnscat_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df.drop(['label','lld'],axis=1).values
Y = df['label'].values
# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)

print("Number Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)




print("[*] Feature Extraction with PCA")
# Feature Extraction with PCA
import numpy
from pandas import read_csv
from sklearn.decomposition import PCA
# load data
df = pd.read_csv('../lld_lab_dnscat_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df.drop(['label','lld'],axis=1).values
Y = df['label'].values
# feature extraction
pca = PCA(n_components=3)
fit = pca.fit(X)
# summarize components
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(fit.components_)


print("[*] Feature Importance with Extra Trees Classifier")
# Feature Importance with Extra Trees Classifier
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
# load data
df = pd.read_csv('../lld_lab_dnscat_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

X = df.drop(['label','lld'],axis=1).values
Y = df['label'].values


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
from sklearn.ensemble import RandomForestClassifier
model_tree = RandomForestClassifier(random_state=100, n_estimators=50)
model_tree.fit(X_train, y_train)
print(model_tree.feature_importances_)
sel_model_tree = SelectFromModel(estimator=model_tree, prefit=True, threshold='mean')
# since we already fit the data, we specify prefit option here
## Features whose importance is greater or equal to the threshold are kept while the others are discarded.
X_train_sfm_tree = sel_model_tree.transform(X_train)
print(sel_model_tree.get_support())
