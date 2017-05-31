#Python 2.7
#Requires scipy to make image

#Since Assignment 4.2 specifically states to use
#numpy, I have refrained from any numpy usage at all in
#this script for 4.1
import sys
import scipy.misc
import time

#USER-INPUT----------------------------------------------
#Global Vars
#Define triangle z=x+iy, x in (a,b) y in (c,d)
if sys.argv.__len__() > 7:
	_a = float(sys.argv[1])
	_b = float(sys.argv[2])
	_c = float(sys.argv[3])
	_d = float(sys.argv[4])
	stepsizex = int(sys.argv[5])
	stepsizey = int(sys.argv[6])
	_filename = sys.argv[7]
else:
	raise ValueError('Usage: mandelbrot_1.py a b c d Nx Ny FILENAME')

_threshold = 1000 #mandelbrot set iteration threshold

print ("Saving image as: {}".format(_filename))

if stepsizex < 0 or stepsizey < 0:
	raise ValueError("Your dimensions are negative!")
#--------------------------------------------------------

#DEFS-----------------------------------------------------
#Mandelbrot sequence tester recursive
#Python is apparently a bad language for recursion..
'''
def f(a,b,x=0,y=0,n=0):
	z_re = x**2-y**2+a
	z_im = 2*x*y+b
	abs_val = (z_re**2+z_im**2)**0.5
	if abs_val<=2 and n<500:
		return f(a,b,z_re,z_im,n+1)
	return n
'''
#Mandelbrot sequence tester iterative
def f(x,y):
	n = 0 #the set always begins with 0, so always apply f once
	c_re = x
	c_im = y
	while(n<_threshold):
		if n == 0:
			z_re = c_re
			z_im = c_im
		else:
			temp = (z_re**2-z_im**2)+c_re
			z_im = 2*z_re*z_im+c_im
			z_re = temp
		if (z_re**2+z_im**2)**0.5-2>=0:
			break
		n=n+1
	return n

#RGB Colour Gradient Function (not used)
'''
def color_gradient(n):
	#n is the iteration score
	ratio = float(_threshold-n)/float(_threshold)
	r = ratio**2*114
	g = ratio**2*82
	b = ratio**2*136
	return r,g,b

#Grid filler for numpy array (not used)
def fill_grid():
	i_max = _grid.shape
	j_max = i_max[1]
	i_max = i_max[0]
	for i in range(0,i_max):
		for j in range(0,j_max):
			_grid[i,j] = f(_x[i],_y[j])
'''
#---------------------------------------------------------------

#Color Grid------------------------------------------------------
#Use colors RGB black:(0,0,0) - white:(255,255,255)
#Fill a grid with colors
#stepsize = 640
#_x = np.linspace(_a,_b,stepsize)
#_y = np.linspace(_c,_d,stepsize)
#_grid = np.zeros((_x.size,_y.size),dtype='float16')
#fill_grid()
start_time = time.time() #calculate time spent creating image
lenx = stepsizex
leny = stepsizey
if stepsizex <= 0:
	lenx = 1
	_b = _a
if stepsizey <= 0:
	leny = 1
	_d = _c
#using pure python, make a grid and fill it
_grid = [ [[] for i in range(0,stepsizex)] for j in range (0,stepsizey)]
for index, item in enumerate(_grid):
	for index2, item2 in enumerate(item):
		x = _a + float(_b-_a)/float(lenx)*index2
		y = _c + float(_d-_c)/float(leny)*index
		item[index2] = f(x,y) #transpose because we are going rowns then columns
print (" %s seconds" %(time.time()-start_time))
#The part that doesnt use "pure python"
#scipy.misc.imshow(_grid)
if stepsizex > 0 and stepsizey > 0:
	scipy.misc.imsave(_filename,_grid)
else:
	print("User Error: Retangle is empty!")
