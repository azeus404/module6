import argparse
import tldextract

"""
Extract domains (LLD) dataset
tokenization
v1
"""

parser = argparse.ArgumentParser(description='Process domainlist')
parser.add_argument('path', help='domainlist')
parser.add_argument('out', help='lldlist')

args = parser.parse_args()
path = args.path
out = args.out

outfile = open(out, "a")
infile = open(path,'r')
outfile.write('lld,\n')
d = infile.readlines()
for i in d:
    e = tldextract.extract(i)
    if e.subdomain:
        print(e.subdomain)
        outfile.write(e.subdomain+'\n')
