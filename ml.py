import pandas as pd
from sklearn.naive_bayes import GaussianNB


data = pd.read_csv('dnsfields_training_labeled.csv')

data.drop('ip', axis=1, inplace=True)
X = data.iloc[:,:-1].values
y = data['malware']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27) 
