#Using: Python 2.7.6, Mar 22, 2014

#Script produces warnings if u are using deprecated numpy API

#Depending on if you have installed weave as standalone or not

#And PLEASE note that the original assignment said to use WEAVE, not CYTHON!! (I had already implemented before this change!)

import numpy as np
import scipy.misc
import sys
import time
import matplotlib.pyplot as plt
#-----------------------------------------------------------------

try:
	from scipy import weave
except:
	try:
		import weave
	except:
		raise ValueError('Please install weave!')


if sys.argv.__len__() > 5:
	_a = float(sys.argv[1])
	_b = float(sys.argv[2])
	_c = float(sys.argv[3])
	_d = float(sys.argv[4])
	_stepsize = int(sys.argv[5])
else:
	raise ValueError('Usage: mandelbrot_3.py a b c d stepsize')


_threshold = 1000
_radius = 4
_filename = "mandelbrot_weave.png"
print ("Storing {}".format(_filename))

#-----------------------------------------------------------------

def mandelbrot(a,b,c,d,stepsize,threshold,radius):
	#we declare an empty image because weave has difficulty returning c types that python understands
	#and let the c-code operate directly on u, i.e. not passing by copy
	#using for-loops that do the same as in mandelbrot_2.py
	#we also make sure to dynamically allocate memory.
	u = np.zeros((_stepsize,_stepsize),dtype=np.uint16);

	code = r"""
	#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
	double _Complex **grid = new double _Complex*[stepsize];
	bool **index = new bool*[stepsize];
	double _Complex **z = new double _Complex*[stepsize];
	for (int i=0; i<stepsize; i++){
		grid[i] = new double _Complex [stepsize];
		index[i] = new bool [stepsize];
		z[i] = new double _Complex [stepsize];
	}


	for(int i=0; i<stepsize; i++){
		for(int j=0; j<stepsize; j++){
			grid[i][j] = a+(b-a)/(stepsize-1)*j + (c+(d-c)/(stepsize-1)*i)*_Complex_I;
			z[i][j] = grid[i][j];
			index[i][j] = true;
		}
	}
	for(int n=0; n<threshold; n++){
		for(int i=0; i<stepsize; i++){
			for(int j=0; j<stepsize;j++){
				if(index[i][j]){
					if( pow(creal(z[i][j]),2)+pow(cimag(z[i][j]),2) <= 4){
						z[i][j] = z[i][j]*z[i][j] + grid[i][j];
						U2(i,j) += 1;
					}else{
						index[i][j] = false;
					}
				}
			}
		}
	}
	for(int i=0; i<stepsize;++i){
		delete[] grid[i];
		delete[] z[i];
		delete[] index[i];
	}
	delete[] grid;
	delete[] z;
	delete[] index;
	"""
	weave.inline(code,['a','b','c','d','stepsize','threshold','radius','u'],compiler='gcc',headers=['<stdbool.h>','<math.h>','<complex.h>'],)
	return u


u = mandelbrot(_a,_b,_c,_d,_stepsize,_threshold,_radius)
print (" %s seconds" %(time.time()-start_time))


#runtime comparison
'''
n = np.log([20,40,60,80,100,120,140,160,320,640,1280,2560,5120])
f = np.log([0.048,0.054,0.063,0.077,0.095,0.121,0.151,0.195,0.672,2.77,9.95,36.5,148.72])
fitted = np.polyfit(n[5:],f[5:],1)
fig, ax1 = plt.subplots(1,1, sharex=False, sharey=False)
plt.plot(n,f,label="mandelbrot_3.py")
textstr = 'Fitted r: %0.2f, c: %0.2f,' %(fitted[0],fitted[1])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
plt.show()
'''

#scipy.misc.imshow(u)
scipy.misc.imsave(_filename,u)