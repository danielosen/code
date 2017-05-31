#RPN Calculator
#Python 2.7
#you can press enter to exit if no other input
import sys
from numpy import sqrt,sin,cos
def isfloat(value):
	try:
		float(value)
		return True
	except:
		return False

def calculator(new_item,stack):
	if True:
		stacksize = stack.__len__()
		if isfloat(new_item):
			stack.append(float(new_item))
			#print("added to stack")
		elif new_item =="p":
			if len(stack) > 0:
				print(stack[-1])

		elif new_item == "*":
			if stacksize>=2:	
				product=(stack[-1])*(stack[-2])
				stack.pop()
				stack.pop()
				stack.append(product)
		elif new_item == "/":
			if stacksize>=2:
				if stack[-1] != 0:
					division=(stack[-2])/(stack[-1])
					stack.pop()
					stack.pop()
					stack.append(division)
				else:
					print("Division by zero ignored")
		elif new_item == "+":
			if stacksize>=2:
				plus=(stack[-1])+(stack[-2])
				stack.pop()
				stack.pop()
				stack.append(plus)
		elif new_item == "sin":
			if stacksize>=1:
				sinus=sin((stack[-1]))
				stack.pop()
				stack.append(sinus)	
		elif new_item == "cos":
			if stacksize>=1:
				cosinus=cos((stack[-1]))
				stack.pop()
				stack.append(cosinus)
		elif new_item == "v":
			if stacksize>=1:
				if stack[-1]>=0:
					squareroot=sqrt((stack[-1]))
					stack.pop()
					stack.append(squareroot)
				else:
					print("squareroot of negative number ignored")
		else:
			print("calculator can't calculate")
			pass

def inputparser(new_items,stack):
	while(new_items):
		for s in new_items:
			calculator(s,stack)
		new_items=raw_input().split()
		

if __name__ == '__main__':
	stack = []
	if len(sys.argv)>1:
		new_items = sys.argv[1].split()
		#print(new_items)
	else:
		new_items = raw_input().split()
		#print(new_items)
	inputparser(new_items,stack)
	

			





