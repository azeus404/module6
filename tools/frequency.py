import argparse
import csv

"""
check frequency of characters in LLD domain names
"""
parser = argparse.ArgumentParser(description="LLD")
parser.add_argument('path',help='Add path')

args=parser.parse_args()
path = args.path

infile = open(path,'r')
#outfile = csv.writer(open("output.csv", "w"))
outfile = open("outfile.txt","w")
data = infile.read().splitlines()

for input_string in data:
    frequency_table={}
    for char in input_string:
        frequency_table[char] = frequency_table.get(char,0)+1

    #show output
    print(input_string, str(frequency_table))
    #outfile.write(input_string +','+str(frequency_table)+'\n')
    #write output
    for key, val in frequency_table.items():
        print(key)
outfile.close()
