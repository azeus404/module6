
# Paper MSCE 2019

Applying Machine learning for detection of data exfiltration using DNS protocol.

This method uses lexical features of DNS names to predict if a domain query is benign or malicious.

Used name conventions: LLD = Lower level domain)
# classification models
- K-Nearest Neighbor (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forrest
- Neural network

# Best performance

# Related work
https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas
https://kldavenport.com/detecting-randomly-generated-domains/
https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn

# Setup & run script
0 - Install the requirements pipenv install -r requirements.txt
start jupyter notebook
1 - Run preprocessing script python ml_datapreprocessing.py lld_lab_data.csv --out lld_lab_features_added.csv
2 - Run the defferent machine learing model scripts different ml_<model>.py lld_lab_features_added.csv

# Features
we used domain name entropy, domain name string length and ratio of aplha numeric characters vs letters in the domain name and number of dot's in domain

## Compare distribution of entropy
We can compare the distribution of entropy for normal vs Malicious domains via histograms and parametric statistics.

# Dataset
We used a dataset RAW_DATA/dnsfields_raw.csv and preprocessed it to lld_labeled.csv

## Lab data
We used a mixed dataset created in our lab.
lld_lab_data.csv

## The detector
We made a proof of concept of the detector as a detector script
python detector.py
