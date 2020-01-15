
# Paper MSCE 2019

Applying Machine learning for detection of data exfiltration using DNS protocol.

This method uses lexical features of DNS names to predict if a domain query is benign or malicious.

Used name conventions: LLD = Lower level domain)
# classification models
We used models for supervised learning:
- K-Nearest Neighbor (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forrest
- Neural network

# Best performance
We use:
# https://muthu.co/understanding-the-classification-report-in-sklearn/
- classification_report
- confusion_matrix
- ROC/AUC

# Related work
https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas
https://kldavenport.com/detecting-randomly-generated-domains/
https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn

# Setup & run script
1. Install the requirements pipenv install -r requirements.txt
start jupyter notebook
2. Run preprocessing script python ml_datapreprocessing.py lld_lab_data.csv --out lld_lab_features_added.csv
3. Run the different machine learning model scripts different ml_<model>.py lld_lab_features_added.csv

# Features
we extracted the DNS subdomain form the testdata and derived some features of this string:
- subdomain entropy
- subdomain string length
- ratio of alpha numeric characters vs letters in the subdomain
- number of dot's in subdomain
- number of unique character in subdomain

We selected these features because tunnel queries are mainly non-human readable of pronounceable.

## Compare distribution of entropy
We can compare the distribution of entropy for normal vs Malicious domains via histograms and parametric statistics.

# Dataset
we used two full network captures of a lab.

## Lab data
We used a mixed dataset created in our lab.
One used DNScat and the other Iodinde for DNS covert channels supplemented it with benign traffic by running a tool called PartyLoud https://github.com/realtho/PartyLoud to create benign DNS traffic

## The detector
We made a proof of concept of the detector python script
python detector.py
