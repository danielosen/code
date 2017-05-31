#highlight parser
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

				#Parse the line,
				#char by char, and keep a memory of the recent match.
				mem = []
				match_mem = []
				l = line.__len__()
				for idx,char in enumerate(line):
					#store the char
					mem.append(char)

					#match syntax
					#keywords
					keyword_list = ["if","elif","else"]
					valid_identifiers = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r"
										"s","t","v","w","x","y","z","_","0","1","2","3","4","5","6","7","8","9"]
					for keyword in keyword_list:
						if mem[-1] == keyword:
							for keyword in keyword_list:
								if mem








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

	


