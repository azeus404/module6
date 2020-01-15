import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
"""
Scatter plot
A Framework for DNS Based Detection and Mitigation of Malware Infections on a Network
"""
df = pd.read_csv('lld_lab_dnscat_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

sns.scatterplot(x="length", y="entropy", hue="label",data=df)

plt.title('LLD Length vs Entropy - DNScat')
plt.show()


sns.distplot(df['entropy'])
sns.distplot(df['length'])
plt.title('LLD Length vs Entropy - DNScat')
plt.show()

df = pd.read_csv('lld_lab_iodine_features_added.csv',encoding='utf-8')
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

sns.scatterplot(x="length", y="entropy", hue="label",data=df)
plt.title('LLD Length vs Entropy - Iodine')
plt.show()

plt.title('LLD Length vs Entropy - Iodine')
sns.distplot(df['entropy'])
sns.distplot(df['length'])
plt.show()
