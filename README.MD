
# Detecting C2 over DNS using LLDs
Paper MSCE 2019

We use supervised machine learning algorithms to detect of data exfiltration using DNS protocol.
This method uses lexical features in lower level domain names (lld) to predict if a domain query is benign or malicious.

# Feature selection
We used features based on the statistics of the lld:
- entropy
- length
- ratio between characters and numbers in the lld

And validated them based on Lasso and RFE (Feature elimination).
In this way we remove features to prevent over fitting the model

# classification models
We used models for supervised learning:
- K-Nearest K-Nearest neighbor (KNN)
- Logistic Regression
- Support Vector Machine (SVM)
- Naive Bayes
- Decision Tree
- Random Forrest (Ensemble learning)
- Neural network

#preprocessing
https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/
We applied different methods for rescaling the dataset:

We shuffled the trainings data to prevent overfitting.

# Hyper parameter Tuning
We applied hyper parameter tuning.

https://www.kaggle.com/mayu0116/hyper-parameters-tuning-of-dtree-rf-svm-knn

The scores of the the tuned models is recored in /scores directory.

# Best performance
We compared the models and used the models with the highs accuracy for the detector.

We selected Random Forrest classifier with accuracy 99.00%
And because of statement in related work, where ensemble learning (yassine,2018) has given the best results.


# https://muthu.co/understanding-the-classification-report-in-sklearn/
- classification_report
- confusion_matrix
- ROC/AUC
- Precision-recall curve is better of unbalanced datasets

# Related work
https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas
https://kldavenport.com/detecting-randomly-generated-domains/
https://www.kaggle.com/amolbhivarkar/knn-for-classification-using-scikit-learn

# Setup & run script
1. Install the requirements pipenv install -r requirements.txt
start jupyter lab
2. extract lld from pcap data
3. Run preprocessing script python ml_datapreprocessing.py lld_lab_data.csv --out lld_lab_features_added.csv
4. Run the different machine learning model scripts with python ml_<model>.py lld_lab_features_added.csv

# Features
we extracted the DNS subdomain form the testdata and derived some features of this string:
- subdomain entropy
- subdomain string length
- ratio of alpha numeric characters vs letters in the subdomain
- number of dot's in subdomain
- number of unique character in subdomain

We selected these features because tunnel queries are mainly non-human readable of pronounceable.

## Compare distribution of entropy
We can compare the distribution of entropy for benign vs Malicious domains via histograms and parametric statistics.

# Dataset
we used two full network captures of a lab.

## Lab data
We used a mixed dataset created in our lab.
One used DNScat and the other Iodine for DNS covert channels supplemented it with benign traffic by running a tool called PartyLoud https://github.com/realtho/PartyLoud to create benign DNS traffic

## The detector
We made a proof of concept of the detector python script
python detector.py
