#mandelbrot_3_cython.py
import numpy as np
import time
import sys
import mandelbrot_cython
import scipy.misc 

if sys.argv.__len__() > 7:
	_a = float(sys.argv[1])
	_b = float(sys.argv[2])
	_c = float(sys.argv[3])
	_d = float(sys.argv[4])
	_nx = int(sys.argv[5])
	_ny = int(sys.argv[6]) 
	_filename = sys.argv[7]
else:
	raise ValueError('Usage: mandelbrot_3.py a b c d Nx Ny FILENAME')
escapetime=1000
print("Saving image as: {}".format(_filename))
if _nx < 0 or _ny < 0:
	raise ValueError("Your dimensions are negative!")

start_time = time.time()

image = np.ones((_ny,_nx),dtype=np.uint16)
mandelbrot_cython.generate_Image(image,_a,_b,_c,_d,_ny,_nx,escapetime)

print (" %s seconds" %(time.time()-start_time))

if _nx != 0 and _ny != 0:
	scipy.misc.imsave(_filename,image)
else:
	print("User Error: Retangle is empty!")