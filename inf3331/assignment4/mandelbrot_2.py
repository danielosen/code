#Python 2.7
#Mandelbrot Set Image optimized for numpy arrays

import numpy as np
import sys
import scipy.misc
#import matplotlib.pyplot as plt
import time
import random
#----------------------------------------------------

if sys.argv.__len__() > 7:
	_a = float(sys.argv[1])
	_b = float(sys.argv[2])
	_c = float(sys.argv[3])
	_d = float(sys.argv[4])
	_nx = int(sys.argv[5])
	_ny = int(sys.argv[6])
	_filename = sys.argv[7] 
else:
	raise ValueError('Usage: mandelbrot_2.py a b c d Nx Ny FILENAME')

_threshold = 1000 #mandelbrot set iteration threshold

#for creating other sets than mandelbrot sets
_radius = 4.0
_power = 2#3
_cube = 0#-0.7198 + 0.9111*1j

print("Saving image as {}".format(_filename,))

#----------------------------------------------------
#Vectorized mandelbrot sequence tester
if _nx < 0 or _ny < 0:
	raise ValueError("Your dimensions are negative!")

start_time = time.time()
y,x = np.ogrid[_c:_d:_ny*1j,_a:_b:_nx*1j] #make two vectors with stepsize number of elements going from start- to endpoint. This actually makes one row and one column vector
c = x+y*1j  #This is an unmathematical operation, since the sum of two matrices of unequal size is not well-defined.
		#However, numpy has no respect for 1D matrices, and so treats this sum (for size 2 etc.) like [1,1]^T * x + y*[1,1] = [x_0+y_0,x_1+y_0;x_0+y_1,x_1+y_1] which is what we need
z = np.zeros(c.shape,dtype=np.complex128) #start the sequence with complex z=0 + 0*i, we use complex64 to free up some memory, although we retain only half precision
index = np.ones(c.shape,dtype=bool) #list of indices to consider at each calculation step, it begins with only True because we consider all points in the beginning
#Make an empty finale image, where we assume every point is not a mandelbrot point
final_image = np.zeros(c.shape,dtype=np.uint16) #we dont really need uint16, but uint8 is too small for _threshold > 255

#s_avg = np.zeros(z.shape,dtype=np.complex64) #this is only used for average color mapping, to sum over an addend function, has nothing to do with actual mandelbrot set

for i in range (0,_threshold):
	z[index] = np.power(z[index],_power)+_cube*z[index] + c[index] #this product is elementwise multiplication, once again treated as arrays and not matrices, we are only concerned with points
											#which haven't failed yet. Clearly, arrays are not matrices in numpy.

	#s_avg[index] += 0.5*np.sin(3.0*np.angle(z[index]))+0.5			#averaging using stripes addend, with s= 3.0
	
	#We now check if any |z|^2 > 4 at step n, in which case we know this c has failed the mandelbrot sequence test
	#We store the positions of the failed c values so we never have to check them again, that is, we "remove" them as indexes to do any calculations with.
	
	index[index] = z[index]*np.conj(z[index]) <= _radius   #This is a small hack, because the RHS has linear index 0....stepsize^2, and Index does not, but by indexing the array
													#with itself, index[index], we can assign the values through this linear index on RHS, similar to MATLAB,
													#and we only update those indices which are still considered. It's exactly what we want!
													#The boolean expression is flipped, because:  z that fail the test return false, hence these indexes must be false,
													#to no longer consider them for further calculation
													#while those that do not fail, return true, hence these indices must be true so that they remain in calculation
	
	#update the final image with any points which are still presumably mandelbrot points, storing at what point in the iteration they have just succeeded
	#Once again, by using the "fancy index" of numpy, we only update points which are still considered.
	final_image[index] += 1
print (" %s seconds" %(time.time()-start_time))

#s_avg = np.divide(s_avg,(final_image+1)) #divide by number of iterations to obtain average (we need to be careful not to divide by 0)

#Make the image
#scipy.misc.imsave("test2.png",final_image)

#Making a color function...
#OVerall, it seems that linear color schemes look very bad,
#Suggested "good" schemes are usually exponential in nature, where color change is most rapid for low escape times
#To make it a bit more scientific, it makes sense to make bands of the so called escape times
#so that we can look and see what areas where the complex number escaped quickly or slowly, grouped by the bands
'''
def to_color(image,z_image,flag):
	#create a new RGB array
    #here we fill the RGB array with rgb values using the grayscale image
    #additional options can produce varied colours by changing the dependency on escape time
    #Using escapetime masking produces decent results
    #so if t=escapetime in [0,4] we use one color, t in [5,something] etc.
    if flag == "basic": #no coloring or smoothing
    	return image
    else:
    	mu = image + 1 - np.log(np.log(np.absolute(z)+1))/np.log(2) #smoothing parameter
    	i,j = image.shape
    	out = np.empty((i,j, 3), dtype=np.uint8) #uint8 is 0-255 which is rgb vals
    	maxmu = np.max(mu)
    	if flag == "ocean":
    		out[:,:,0] = np.uint8(50*np.sin(0.08*mu)**2)
    		out[:,:,1] = np.uint8(100*np.sin(0.04*mu)**2)
    		out[:,:,2] = np.uint8(255*np.sin(0.04*mu)**2)
    	elif flag == "acid":
    		out[:,:,0] = np.uint8(255*np.sin(3*mu)**2)
    		out[:,:,1] = np.uint8(255*np.sin(8*mu)**2)
    		out[:,:,2] = np.uint8(255*np.sin(6*mu)**2)
    	elif flag == "smooth":
    		out[:,:,0] = np.uint8(255*mu/maxmu)
    		out[:,:,1] = out[:,:,0]
    		out[:,:,2] = out[:,:,0]
    	else: #random coloring
    		out[:,:,0] = np.uint8(random.randint(0,255)*mu/maxmu)#*(mu/maxmu*s_avg))
    		out[:,:,1] = np.uint8(random.randint(0,255)*mu/maxmu)#*(mu/maxmu*s_avg))
    		out[:,:,2] = np.uint8(random.randint(0.255)*mu/maxmu)#*(mu/maxmu*s_avg))
    	return out
'''
#final_image = to_color(final_image,z,_e)
#scipy.misc.imshow(final_image)
#plt.imshow(final_image,cmap="hot")
#plt.show()
if _nx > 0 and _ny > 0:
	scipy.misc.imsave(_filename,final_image)
else:
	print("User Error: Retangle is empty!")
#print(final_image)

#NOT USED, SEE REPORT
#plotting results from runtime analysis 
'''
array_steps = np.log(np.array([20,40,60,80,100,120,140,160,320,640]))
array_time2 = np.log(np.array([0.1,0.2,0.3,0.5,0.7,1.0,1.3,1.7,7.3,27.4]))
array_time1 = np.log(np.array([0.3,1.4,3.0,5.3,8.2,11.7,16.0,21.0,83.5,333]))
array_diff = array_time1 - array_time2
fitted1 = np.polyfit(array_steps[4:],array_time1[4:],1)
fitted2 = np.polyfit(array_steps[4:],array_time2[4:],1)
fitted3 = np.polyfit(array_steps[4:],array_diff[4:],1)
fig, ax1 = plt.subplots(1,1, sharex=False, sharey=False)
plt.plot(array_steps,array_time1,label="mandelbrot_1.py")
plt.plot(array_steps,array_time2,label="mandelbrot_2,py")
plt.plot(array_steps,array_diff,label="difference")
textstr = 'Fitted And ExpDiff: %0.2f, %0.2f,%0.2f' %(fitted1[1],fitted2[1],fitted3[1])
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14, verticalalignment='top', bbox=props)
plt.show()
'''

