#Python 2.7
#SWIG Exercise 4.4
#mandelbrot_4.py
import numpy as np
import ctypes as c
#Using SWIG requires a wrapper file mandelbrot_generator.i
#this file has to be in the same folder!!
try:
	import mandelbrot_generator
except:
	raise ValueError("Please read mandelbrot_generator.i!")

import sys
import time
import scipy.misc

if sys.argv.__len__() > 7:
	_a = float(sys.argv[1])
	_b = float(sys.argv[2])
	_c = float(sys.argv[3])
	_d = float(sys.argv[4])
	_sizex = int(sys.argv[5])
	_sizey =int(sys.argv[6])
	_filename = sys.argv[7]
else:
	raise ValueError('Usage: mandelbrot_3.py a b c d Nx Ny FILENAME')

print("Saving image as {}".format(_filename))

escapetime = 1000

if _sizex < 0 or _sizey < 0:
	raise ValueError("Your dimensions are negative!")

start_time = time.time()

image = np.zeros((_sizey,_sizex),dtype=np.uint16)
mandelbrot_generator.generate_Image(image,_a,_b,_c,_d,escapetime) #void function in place modification of image, dims of image is passed with swig typemap

print (" %s seconds" %(time.time()-start_time))

if _sizex > 0 and _sizey > 0:
	scipy.misc.imsave(_filename,image)
else:
	print("User Error: Retangle is empty!")