#test_mandelbrot.py

#Import our swix module for computation, as this was revealed to be the fastest when having variable escapetime
import mandelbrot_generator as mg

#Import numpy for arrays
import numpy as np

#Import scipy for plotting
import scipy.misc

#For testing:
#sudo pip3 install pytest
#Guide: https://syndbg.wordpress.com/2014/04/20/python-3-how-to-write-unit-tests-unittest-pytest-nose/
#py.test-3 test_mandelbrot.py
#Results: 2/2 passed

#METHODS------------------------------------------------------------------------------------------------------

def test_outside():
	image = compute_mandelbrot(3,4,3,4,100,100,1)
	assert(not np.any(image))

def test_inside():
	image = compute_mandelbrot(-0.1,0.1,-0.1,0.1,100,100)
	image = (image == 1000)
	assert(np.all(image))

def compute_mandelbrot(xmin,xmax,ymin,ymax,Nx,Ny,max_escape_time=1000,plot_filename=None):
	
	#Allocate image
	image = np.empty((Ny,Nx),dtype=np.uint16)

	#Check to see if rectangle has at least one point in mandelbrot set first
	mg.generate_Image(image,xmin,xmax,ymin,ymax,1)
	if np.any(image):
		#compute up to escape time since a point was found (image is re-zeroed in function)
		mg.generate_Image(image,xmin,xmax,ymin,ymax,max_escape_time)

	#Plot image if user wants
	if plot_filename:
		plot_filename="{}.png".format(plot_filename)
		scipy.misc.imsave(plot_filename,image)
		print("Saving image as {}".format(plot_filename))

	#Return image
	return image

def color_image(xmin,xmax,ymin,ymax,Nx,Ny,color_scheme="basic",color_region="outer",max_escape_time=1000,plot_filename="mandelbrot"):

	#Grab image
	image = compute_mandelbrot(xmin,xmax,ymin,ymax,Nx,Ny,max_escape_time)

	#Create smoother coloring scheme using information about complex rectangle
	Ny,Nx = np.shape(image)
	y,x = np.ogrid[ymin:ymax:Ny*1j,xmin:xmax:Nx*1j]
	c = x+y*1j
	mu = image + 1 - np.log(np.log(np.absolute(c)+1))/np.log(2)**2
	maxmu = np.max(mu)

	#Allocate proper RGB image
	out = np.empty((Ny,Nx, 3), dtype=np.uint8)

	#Do some additional cool stuff
	if color_region == "inner":
		mu[image < max_escape_time]  = 0
	elif color_region == "outer":
		mu[image == max_escape_time] = 0

	s = 0.5+0.5*(max_iteration_time)/1000
	#Begin coloring RGB
	if color_scheme == "ocean":
		out[:,:,0] = np.uint8(50*np.sin(s*0.08*mu)**2)
		out[:,:,1] = np.uint8(100*np.sin(s*0.04*mu)**2)
		out[:,:,2] = np.uint8(255*np.sin(s*0.04*mu)**2)
	elif color_scheme == "acid":
		out[:,:,0] = np.uint8(255*np.sin(s*3*mu)**2)
		out[:,:,1] = np.uint8(255*np.sin(s*8*mu)**2)
		out[:,:,2] = np.uint8(255*np.sin(s*6*mu)**2)
	else:
		out[:,:,0] = np.uint8(255*mu/maxmu)
		out[:,:,1] = out[:,:,0]
		out[:,:,2] = out[:,:,0]

    #Plot
	plot_filename="{}_{}_{}.png".format(plot_filename,color_scheme,max_escape_time)
	scipy.misc.imsave(plot_filename,out)
	print("Saving image as {}".format(plot_filename))
	pass


'''
if __name__ == '__main__':
	pass
'''