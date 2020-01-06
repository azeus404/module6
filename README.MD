
# Paper MSCE 2019

Using Machine learning for detection of data exfiltration using DNS protocol over a covert channel
Used name conventions: LLD = Lower level domain)

# Related work
https://kldavenport.com/detecting-randomly-generated-domains/

# Setup & run script
Install the requirements
Run python ml.py
exports a CSV: lld_features_added.csv

# Features
we used domain entropy and string length

## Compare distribution of entropy
We can compare the distribution of entropy for normal vs Malicious domains via histograms and parametric statistics.


# Dataset
We used a dataset RAW_DATA/dnsfields_raw.csv and preprocessed it to lld_labeled.csv
Use the command line to label dataset sed 's/$/,0/' lld_mm.txt >> lld_mm_labeled.txt

## Lab data
We used a mixed dataset created in our lab.
lld_lab_mixed.txt

### Non-Malicious
we used a list of Non-Malicious domains form majestic.
https://blog.majestic.com/development/majestic-million-csv-daily/

### Malicious
we used a list of malicious domains from the sans institute.
https://isc.sans.edu/feeds/suspiciousdomains_High.txt
