import argparse
import tldextract

"""
Extract domains (LLD) dataset
tokenization
https://www.endgame.com/blog/technical-blog/using-deep-learning-detect-dgas
v1
"""

parser = argparse.ArgumentParser(description='Process domainlist')
parser.add_argument('path', help='an integer for the accumulator')


args = parser.parse_args()
path = args.path

file = open(path,'r')
d = file.readlines()
for i in d:
    e = tldextract.extract(i)
    #print(e.domain)
    if e.subdomain:
        print(e.subdomain)
    #print(e.fqdn)
