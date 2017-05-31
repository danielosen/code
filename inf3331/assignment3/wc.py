#wc.py
#python 2.7
#apparently script scope is limited to .txt files and .py files only according to instructor on piazza
import sys
#current directory path as string
currentpath = sys.path[0]
#Assume filenames are separated with whitespace
filenames = sys.argv[1:]
#number of lines
totlines = 0
totwords = 0
totchars = 0
for fn in filenames:
	with open(fn, "r") as f:
		numlines = 0
		numwords = 0
		numchars = 0
		lines = f.readlines()
		for line in lines:
			numchars += line.__len__() #this counts formatting characters just like wc does
			numwords += len(line.split()) #splits on all whitespaces, regardless if made by indentation
			if "\n" in line:
				numlines+=1
		print numlines,"\t",numwords,"\t",numchars,"\t",fn
		totlines += numlines
		totwords += numwords
		totchars += numchars
print totlines,"\t",totwords,"\t",totchars,"\t","total"
