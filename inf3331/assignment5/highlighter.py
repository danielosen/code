#highlighter.py
'''Takes in syntaxfile, themefile, and sourcefile. Outputs colored text from sourcefile to standard out'''

import sys,re

def color_substring(color_code,substring):
	'''Colors a substring using bash color sequences'''

	start_code = "\033[{}m".format(color_code)
	end_code = "\033[0m"
	return str(start_code+substring+end_code)

def print_error(error_code):
	'''Prints customized colored errors'''

	if error_code == "args":
		print("\n"+color_substring(31,"ERROR: ")+"MISSING ARG(S)\n")
		print(color_substring(36,"USAGE: ")+"python3 highlighter.py "+color_substring(92,"syntaxfile themefile sourcefile_to_color")+"\n")
		print(color_substring(33,"NB: ")+"These are given as "+color_substring(92,"filename.file_extension")+", located in script folder.\n")
	
	elif error_code == "notfound":
		print("\n"+color_substring(31,"ERROR: ")+"FILE(S) NOT FOUND\n")
	
	else:
		pass


def make_syntax_dict(syntaxfile_name):
	'''creates a regular expression/syntax dictionary for matching expressions to classes'''

	try:
		with open(syntaxfile_name,'r') as syntaxfile:
			syntax_dict = {}
			comment_dict = {}
			string_dict = {}
			for line in syntaxfile:
				#line = line.strip
				#print(line)
				match_obj = re.match("\"(.*)\":\s(.*)\s?",line)
				if bool(match_obj):
					#print(match_obj)
					#There are three distinct categories: code, strings and comments
					#we do not implement multiline comments/multidocstrings
					if match_obj.group(2) == 'comment':
						comment_dict[match_obj.group(1)] = match_obj.group(2)
					elif match_obj.group(2) == 'string' or match_obj.group(2) == 'docstring':
						string_dict[match_obj.group(1)] = match_obj.group(2)
					else: 
						syntax_dict[match_obj.group(1)] = match_obj.group(2)
			return comment_dict,string_dict,syntax_dict

	except FileNotFoundError:
		print_error("notfound")
	
	else:
		pass

def make_theme_dict(themefile_name):
	'''creates a dictionary of matching classes to colors'''

	try:
		with open(themefile_name,'r') as themefile:
			theme_dict = {}
			for line in themefile:
				line = line.strip()
				match_obj = re.match("(.*):\s(.*)\s?",line)
				if bool(match_obj):
					theme_dict[match_obj.group(1)] = match_obj.group(2)
			return theme_dict
	
	except FileNotFoundError:
		print_error("notfound")
	
	else:
		pass


def highlight_text(syntaxfile_name,themefile_name,sourcefile_to_color_name):
	'''highlights the sourcefile text with syntaxfile's regexp rules and themefile's color rules'''


	#Grab Dictionaries (comments are separated out because they have special formatting rules)
	comment_dict,string_dict,regex_dict = make_syntax_dict(syntaxfile_name)
	theme_dict = make_theme_dict(themefile_name)

	#Create list of modifications, separate comments/strings here as well
	modification_list = []
	comment_list = []
	docstring_list = []

	#Iterate over lines in file and color them

	try:
		with open(sourcefile_to_color_name,'r') as sourcefile_to_color:
			#open the file, iterate line by line, and iterate through dictionaries to find regex matches
			for line in sourcefile_to_color.readlines():
				for key,value in regex_dict.items():
					#get regex matches syntax
					#re.finditer(key,line) returns an iterator yielding all non-overlapping match_objects with pattern key, in string line
					array_of_match_obj = re.finditer(key,line)
					for match_obj in array_of_match_obj:
						#restrict to innermost group per match
						idx = match_obj.lastindex
						if idx is None: #no groups
							idx = 0
						#print(match_obj.group(idx))
						modification_list.append((match_obj.start(idx),match_obj.end(idx),theme_dict[value]))
						#It appears as if the match obj contains at most one match
						

				#get regex matches comments
				#There are at least two comment types
				#Full line comments, i.e. everything to right of symbol, and enclosed comments.
				#multiline is not implemented
				for key,value in comment_dict.items():
					array_of_match_obj = re.finditer(key,line)
					for match_obj in array_of_match_obj:
						idx = match_obj.lastindex
						if idx is None:
							idx = 0
						comment_list.append((match_obj.start(0),key,theme_dict[value]))
						#print(match_obj)
						#there is no need to grab index, since comments are full match anyway

				#get regex matches strings
				for key,value in string_dict.items():
					array_of_match_obj = re.finditer(key,line)
					for match_obj in array_of_match_obj:
						idx = match_obj.lastindex
						if idx is None: #no groups
							idx = 0
						#print(match_obj.group(idx))
						#strings are also full match, however we want to be able to get more than one string per line
						#even if they are the same type of string
						docstring_list.append((match_obj.start(idx),match_obj.end(idx),theme_dict[value]))

				#Sort lists and prepare color stacks
				#also have some default vals if list is empty
				l = line.__len__()
				#this is a backup line for removing formatting
				#not really needed, but easier then regexing away previous formatting
				linecopy = line

				if comment_list:
					comment_list.sort()
					cs_i,ckey_i,cc_i = comment_list.pop()
				else:
					cs_i = -999

				if docstring_list:
					docstring_list.sort()
					ss_i,se_i,sc_i = docstring_list.pop()
				else:
					ss_i = -999
					se_i = l

				if modification_list:
					modification_list.sort()
					#print(modification_list)
					s_i,e_i,c_i = modification_list.pop()
				else:
					s_i= -999
				for i in range(l,-1,-1):

					#Explanation:
					#Beginning from the end of the string, we iterate character by character, going right to left.
					#At each character position in the string, we check if this position is equal to any
					#starting position of a matched regex position (match.start).
					#If it is equal to a matched expression, then we insert the color formatting at this position,
					#otherwise we simply print the character as is. 

					#The reason we iterate from right to the left, is because then we achieve that:
					#any indices up to newest changed substring are still valid, even when the string has been modified.

					#The real challenge is handling rules for determining how regex conflicts interact...

					#At a matching position, we first match with the "strings" category. If we do match with this category
					#then in the interest of avoiding conflicts, we do not match anything else at this position.
					#We also "ban" non-string non-comment matches from entering this modified substring, because
					#inside strings we wish to avoid all other formatting. The reason we match strings first is also because of this.
					#Avoiding all formatting inside strings when highlighting is very common, except for a small subset of special cases.

					#In the strings category there is a single corner case in Python. This is because docstrings and
					#the regex for ordinary strings have the same starting character. When this happens, we treat
					#the docstring as a comment. This occurs in other languages too

					#The next matching category is comments. These are single line comments like this line, which block
					#absolutely any formatting to the right of it, except if found inside a string (hence strings block and are first).
					#If a valid comment symbol is found before a valid string, then we ignore the previous formatting, hence the copy.
					#we also block other syntaxing from entering to the right of the comment.

					#the last category is any non-string non-comment.

					if i==ss_i:
						#print(ss_i,se_i,sc_i,l)
						if i==cs_i and cs_i!= -999:
							#color it as comment instead
							line = linecopy[:i]+color_substring(cc_i,linecopy[i:])
						else:
							line = line[:i]+color_substring(sc_i,line[i:se_i])+line[se_i:]
						if docstring_list:
							ss_i,se_i,sc_i = docstring_list.pop()
					elif i==cs_i:
						if i not in range(ss_i,se_i) or ss_i == -999: #dont print from inside string
							#if we actually do print, disregard all previous formatting to the right
							line = linecopy[:i]+color_substring(cc_i,linecopy[i:])
						if comment_list:
							cs_i,ckey_i,cc_i = comment_list.pop()
					elif i == s_i:
						#print("<> ",s_i,e_i,c_i,l)
						if i not in range(cs_i,l+1) or cs_i==-999: #dont print to the right of comment
							if i not in range(ss_i,se_i) or ss_i==-999: #dont print inside string
								#print(" printed!")
								line = line[:i]+color_substring(c_i,line[i:e_i])+line[e_i:]
						if modification_list:
							s_i,e_i,c_i = modification_list.pop()
					else:
						pass

				print(line)

	except FileNotFoundError:
		print_error("notfound")
		
	else:
		pass

if __name__ == "__main__":
	'''Executs the script on the filename arguments'''

	try:
		syntaxfile_name = sys.argv[1]
		themefile_name = sys.argv[2]
		sourcefile_to_color_name = sys.argv[3]
		highlight_text(syntaxfile_name,themefile_name,sourcefile_to_color_name)

	except IndexError:
		print_error("args")

	#else:
	#	pass

	