import argparse

"""
check frequency of characters in LLD domain names
"""
parser = argparse.ArgumentParser(description="LLD")
parser.add_argument('path',help='Add path')

args=parser.parse_args()
path = args.path

file = open(path,'r')
d = file.read().splitlines()

for input_string in d:
    frequency_table={}
    for char in input_string:
        frequency_table[char] = frequency_table.get(char,0)+1

    #show output
    print(input_string, str(frequency_table))
