import argparse
"""
Extract n-grams from LLD dataset
"""
parser = argparse.ArgumentParser(description="LLD")
parser.add_argument('path',help='Add path')
parser.add_argument('grams',type=int,help='number of gram', default=2)

args = parser.parse_args()
path = args.path
gram = args.grams

infile = open(path,'r')
data = infile.read().splitlines()
for test_list in data:
    n = gram
    print([test_list[i:i+n] for i in range(0, len(test_list), n)])
    #print(test_list)
