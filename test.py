import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

print('[+] Feature Importance Lasso')
# load data
df = pd.read_csv('../lld_lab_dnscat_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
numerics = ['int16','int32','int64','float16','float32','float64']
numerical_vars = list(df.select_dtypes(include=numerics).columns)
data = df[numerical_vars]
data.shape

X_train, X_test, y_train, y_test = train_test_split(
    data.drop(labels=['label', 'ID'], axis=1),
    data['target'],
    test_size=0.3,
    random_state=0)

X_train.shape, X_test.shape

scaler = StandardScaler()
scaler.fit(X_train.fillna(0))

sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
sel_.fit(scaler.transform(X_train.fillna(0)), y_train)

sel_.get_support()

selected_feat = X_train.columns[(sel_.get_support())]
print('total features: {}'.format((X_train.shape[1])))
print('selected features: {}'.format(len(selected_feat)))
print('features with coefficients shrank to zero: {}'.format(
      np.sum(sel_.estimator_.coef_ == 0)))
