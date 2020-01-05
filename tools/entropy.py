import numpy as np
from collections import Counter
"""
Calculate entropy
"""
# Remember mixed standard python functions and numpy functions are very slow
def calcEntropy(x):
    p, lens = Counter(x), np.float(len(x))
    return -np.sum( count/lens * np.log2(count/lens) for count in p.values())

infile = open('lld.txt','r')
data = infile.read().splitlines()
for input_string in data:
    print(input_string,calcEntropy(input_string))
