# Source: https://github.com/bhanratt/ICD9CMtoICD10CM
#Brian Hanratty
#bhanratt@asu.edu
#Parses target file with list of 1 or more ICD-9-CM Codes in decimal format (e.g. 410.90) and converts to ICD-10-CM and adds description. 
#Will attempt to match some if decimals are missing but results are not guaranteed! :)

import sys, argparse

data_table = {}
ap = argparse.ArgumentParser()
ap.add_argument("infile",help = "File with list of 1 or more ICD-9 Codes in decimal format (e.g. 410.90). Will attempt to match some if decimals are missing but results are not guaranteed! :)", type=str)
args = ap.parse_args()
print("Brian's Super Simple ICD-9-CM to ICD-10-CM converter!")
#Import ICD 9 to 10 conversion dictionary
try:
	f1=open('icd9to10dictionary.txt')
	header=f1.readline()
	for line in f1:
	    nine = str.strip(line.split('|')[0])
	    ten = str.strip(line.split('|')[1])
	    desc = str.strip(line.split('|')[2])
	    data_table[nine] = [ten,desc]
#Missing dictionary handling
except FileNotFoundError:
	print("Missing dependency: icd9to10dictionary.txt")
	sys.exit(1)

#Import file to convert. Must be in decimal format.
f=open(args.infile)
header=f.readline()
count=0
total=0
fw =open(args.infile+'.out','w')
#header
fw.write('icd9'+'\t'+'icd10'+'\t'+'description')
fw.write('\n')

for line in f:
	total+=1
	stripped=str.strip(line)
	if stripped in data_table:
		count+=1
		fw.write(stripped+'\t' + '\t'.join(data_table[stripped]))
		fw.write('\n')
	else:
		fw.write(stripped+'\t'+'NA'+'\t'+'NA')
		fw.write('\n')

print('Matched '+str(count)+' codes from your list of '+str(total)+' codes')
print("If conversion coverage was poor or didn't work at all try running 'python icd9to10.py -h' to make sure your input is formatted properly.")
f.close()
f1.close()
fw.close()