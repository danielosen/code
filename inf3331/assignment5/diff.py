#diff.py
'''Locate and print differences between two text files'''
import os,re,sys

#Diffing is essentially the longest common subsequence problem,
#though the difflib implementation uses contiguous subsequences.

#the largest problem is actually how to deal with larger files
#a naive solution is to just split the files into smaller files
#though the for line in file is already a lazy generator
#in which case we just need to set some limits on
#what we feel is the maximum 'safe' size of a sequence of
#lines in two files.

#We thus limit ourselves to comparing two subsequences of at most MAXLINE lines
# at a time.


def lcs_table(orig_seq,modi_seq):
	''' the longest common sequence table generator, maximum MAXLINE X MAXLINE'''
	orig_len = len(orig_seq)
	modi_len = len(modi_seq)
	mysum = 0

	#make the table
	table = [[0] * (modi_len+1) for _ in range(orig_len+1)]

	#fill the table
	for i in range(1,orig_len+1):
		for j in range(1,modi_len+1):
			if orig_seq[i-1] == modi_seq[j-1]:

				#if the sequences so far are of equal length
				table[i][j] = table[i-1][j-1] + 1

			else:
				#get the longest sequence
				table[i][j] = max(table[i][j-1], table[i-1][j])
			mysum += 1
	return table,orig_len,modi_len

def write_batch(table, orig_seq, modi_seq, orig_len, modi_len,diff_file):
	'''recursive write-to-file function matching sequences of lines'''

	#if we are not done yet, and the sequences are similar
	if orig_len > 0 and modi_len > 0 and orig_seq[orig_len-1] == modi_seq[modi_len-1]:
		write_batch(table, orig_seq, modi_seq, orig_len-1, modi_len-1,diff_file)
		diff_file.write("0 " + orig_seq[orig_len-1])
		diff_file.write("\n")
	else:
		#here we have run out of either sequence or they are not similar
		#in which case we look for the longest subsequence belonging to either modi_seq or orig_seq
		if modi_len > 0 and (orig_len == 0 or table[orig_len][modi_len-1] >= table[orig_len-1][modi_len]):
			#solve the subproblem and write insertion 
			write_batch(table, orig_seq, modi_seq, orig_len, modi_len-1,diff_file)
			diff_file.write("+ " + modi_seq[modi_len-1])
			diff_file.write("\n")
		elif orig_len > 0 and (modi_len == 0 or table[orig_len][modi_len-1] < table[orig_len-1][modi_len]):
			#solve the subproblem and write deletion
			write_batch(table, orig_seq, modi_seq, orig_len-1, modi_len,diff_file)
			diff_file.write("- " + orig_seq[orig_len-1])
			diff_file.write("\n")

def get_batch(orig_file,modi_file, orig_ptr = 0, modi_ptr = 0,MAXLINE=500):
	'''get the next batch of sequences of MAXLINES strings'''

	orig_seq = []
	modi_seq = []

	orig_file.seek(orig_ptr)
	modi_file.seek(modi_ptr)

	counter = 0
	while(counter<MAXLINE):
		counter += 1
		#readline always returns an empty string it eof is reached
		nextline = orig_file.readline()
		if nextline != "":
			orig_seq.append(nextline.rstrip()) #remove trailing whitespace
		else:
			break
	counter = 0
	while(counter<MAXLINE):
		counter+=1
		nextline = modi_file.readline()
		if nextline != "":
			modi_seq.append(nextline.rstrip())
		else:
			break

	return orig_seq,modi_seq,orig_file.tell(),modi_file.tell()


def process_batch(orig_seq, modi_seq,diff_file):

	if not orig_seq and not modi_seq:
		return False
	else:
		#if either file has run empty, this too is handled by the table and print_batch
		#For both it is no point since we are done with both files
		table,orig_len,modi_len = lcs_table(orig_seq,modi_seq)
		write_batch(table,orig_seq,modi_seq,orig_len,modi_len,diff_file)
		return True

def write_diff(orig_file,modi_file):
	'''solves the lcs problem and writes to a diff_file'''
	#ab mode because of windows/linux disagreement on
	with open('diff_output.txt','w') as diff_file:
		#grab the first batch
		orig_seq,modi_seq,orig_ptr,modi_ptr = get_batch(orig_file,modi_file)
		#process batches 
		while process_batch(orig_seq,modi_seq,diff_file):
			orig_seq,modi_seq,orig_ptr,modi_ptr = get_batch(orig_file,modi_file,orig_ptr,modi_ptr)


if __name__ == '__main__':
	'''Finds the diff between two texts, writes the diff to a txt file, 
	and prints highlighted text to stdout.'''
	try:
		orig_file_name = sys.argv[1]
		modi_file_name= sys.argv[2]
	except IndexError:
		print("Usage: python3 diff.py orig_file modi_file")
	else:
		if sys.argv.__len__() == 0:
			print("Usage: python3 diff.py orig_file modi_file")


	try:
		with open(orig_file_name,'r') as orig_file:
			with open(modi_file_name,'r') as modi_file:
				write_diff(orig_file,modi_file)
				os.system("python3 highlighter.py diff.syntax diff.theme diff_output.txt")

	except FileNotFoundError:
		print("Unable to open file(s)")
