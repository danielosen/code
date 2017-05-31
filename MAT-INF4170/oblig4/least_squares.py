import numpy as np
#import plotly.plotly as py
#import plotly.graph_objs as go
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv

def binary_find_mu(x,t):
	'''returns -1 for the endpoint case'''
	first = 0
	last = t.size-2
	found = False
	mu = -1

	while first <= last and not found:
		midpoint = (first+last)//2
		if t[midpoint] <= x and x < t[midpoint+1]:
			mu = midpoint
			found = True
		else:
			if x < t[midpoint]:
				last = midpoint - 1 
			else:
				first = midpoint + 1 
	return mu


def alg220(p,t,c_0,x):
	'''implements algorithm 2.20
	I must admit I do not fully understand the indexing where the algorithm assumes 
	that c_0 has elements from mu-p to mu for a given mu. I doesnt seem to make sense
	for j <=1. However, i think the last exercise 2.19 clarifies this point a bit.

	As far as memory management goes, we presumably cannot change c_0,
	so we make a deep copy using numpy. Now for each k-loop, the new vector c
	is of one size less. This means we can reduce the size of c at each step.
	Whether or not this reduction is particularly efficient I am not sure, 
	dynamically sized arrays may require re-allocating memory.
	'''
	mu = binary_find_mu(x,t)

	if mu-p+1 == 0:
		raise ValueError('p is too high')
	if mu == -1:
		'''x not in any t_mu <= x < t_{mu+1}'''
		print('endpoint reached')
		return c_0[-1]

	c = np.copy(c_0)
	m = 0
	for k in range(p,0,-1):
		for j in range(mu,mu-k+1-1,-1):
			c[j-m] = np.divide(t[j+k]-x,t[j+k]-t[j])*c[j-1-m] + np.divide(x-t[j],t[j+k]-t[j])*c[j-m]
		c = c[1:]
		m+=1
	return c[mu-m]

def read_data(filename):
	'''reads a .dat file and outputs arrays of x,y,z data
	I couldnt figure out the end condition so i skip the last point'''
	with open(filename,'r') as f:
		reader = csv.reader(f,delimiter=' ')
		x = []
		y = []
		z = []
		first = True
		for row in reader:
			x.append(float(row[0]))
			y.append(float(row[1]))
			if first:
				z.append(float(row[2]))
				first=False
	x = np.asarray(x,dtype=float)
	y = np.asarray(y,dtype=float)
	z = np.asarray(z,dtype=float)
	return x,y,z

def process_data(x,y,ratio):
	'''
	Computes chord length parametrization u[:] and knot vector tau[:] for cubic spline. Assumes x,y,z same size with z const.
	Choice of knot vector: tau[0]=u[0] and tau[-1]=u[-1], and evenly spaced between, length approx ratio% size of number of data points.

	'''
	m = x.size
	u = np.zeros(m,dtype=float)
	u[0] = 0.0
	for i in xrange(1,m):
		u[i] = u[i-1] + np.sqrt( (x[i]-x[i-1])**2.0 + (y[i]-y[i-1])**2.0 )
	p = 3
	n = int(np.round(ratio*m))
	#p+1 regular
	tau = np.linspace(start=u[1],stop=u[-2],num=n,endpoint=True,dtype=float)
	tau = np.concatenate((u[0]*np.ones(p+1),tau))
	tau = np.concatenate((tau,u[-1]*np.ones(p+1),tau))
	#tau = np.linspace(start=u[0],stop=u[-2],num=n+p+1,endpoint=True,dtype=float)
	return u,tau,n,m

def cubic_bspline_eval(j,u_i,tau,n,i,m):
	'''evaluates the cubic B-spline at u_i: B_{3,j,tau_j,...tau_j+p+1}(u_i)
	
	c, memory and overwriting layout at each step, b_j,p is the j'th b_spline of order p

	  l=0      l=1       l=2       l=3
	[ b_j_0 | b_j+1_0 | b_j+2_0 | b_j+3_0 ] k=0
	[ x     | b_j,1   | b_j+1,1 | b_j+2,1 ] k=1
	[ x     | x       | b_j,2   | b_j+1,2 ] k=2
	[ x     | x       | x       | b_j,3   ] k=3

	I think there is a bug here? Cannot find it
	'''
	c = np.zeros(4,dtype=float)
	#zero-order b-splines (i.e. find mu abbreviated)
	for l in xrange(3,-1,-1): #3,2,1,0
		if tau[j+l] <= u_i and u_i < tau[j+l+1]: #tau_[j+4],...,tau_[j] 
			c[l] = 1.0
	#remaining b-splines
	for k in xrange(1,4): #1,2,3
		for l in xrange(3,-1+k): #3,2,1 - 3,2 - 3
			c[l] = np.divide(u_i - tau[j+(l-k)],tau[j+l]-tau[j+(l-k)]) * c[l-1]   +   np.divide(tau[j+(l+1)]-u_i,tau[j+(l+1)]-tau[j+(l-k+1)]) * c[l]
	return c[-1]

def lstsq_cubic(filename,ratio,verbose):
	'''
	Computes the cubic spline least squares approximation to data and plots the result.
	'''
	x,y,z = read_data(filename)
	u,tau,n,m = process_data(x,y,ratio)
	A = np.zeros((m,n),dtype=float)
	for i in xrange(m):
		for j in xrange(n):
			A[i,j] = np.asscalar(cubic_bspline_eval(j,u[i],tau,n,i,m))
	#p+1 regular
	A[0,0] = 1.0
	A[-1,-1] = 1.0
	N = np.dot(A.T,A)
	rank = np.linalg.matrix_rank(N)
	if rank<n:
		c_x,_,_,_ = np.linalg.lstsq(A,x)
		c_y,_,_,_ = np.linalg.lstsq(A,y)
	else:
		c_x = np.linalg.solve(N,np.dot(A.T,x))
		c_y = np.linalg.solve(N,np.dot(A.T,y))
	if verbose:
		print "Rank of A^TA = {}, n= {}".format(rank,n)
		print c_x
		print c_y
	return A,c_x,c_y,tau,m,n,u,x,y,z

def refine(c_x,c_y,tau,n,u,p=3):
	'''
	Here we do the trick by adding 0s to the ends of the coefficients, regularize our knot vector,
	to prepare for using alg220 in the same spline, but in the slightly larger spline space.
	We add p+1 knots at the ends assuming tau[0]=u[0]<u[1] and tau[-1]=u[-1]>u[-2] in parametrization.
	when we do this, our original n increases by 2*(p+1), so we add p+1 0 spline coefficients at each end,
	ensuring that the spline lies in both spaces
	'''
	tau_ref = np.copy(tau)
	c_x_ref = np.copy(c_x)
	c_y_ref = np.copy(c_y)
	left = np.ones(p+1)*(u[0] - np.abs(u[1]-u[0]))
	right = np.ones(p+1)*(u[-1] + np.abs(u[-1]-u[-2]))
	tau_ref = np.concatenate((left,tau_ref))
	tau_ref = np.concatenate((tau_ref,right))
	zeros = np.zeros(p+1)
	c_x_ref = np.concatenate((zeros,c_x_ref))
	c_x_ref = np.concatenate((c_x_ref,zeros))
	c_y_ref = np.concatenate((zeros,c_y_ref))
	c_y_ref = np.concatenate((c_y_ref,zeros))
	return c_x_ref,c_y_ref,tau_ref

def refine2(c,p):
	c_ref = np.copy(c)
	left = np.zeros(p+1)
	right = np.zeros(p+1)+c[0]
	c_ref = np.concatenate((left,c))
	c_ref = np.concatenate((c,right))
	return c_ref


def showcase():
	'''show least squares approximations'''
	ratio = 0.2
	force_regular = True
	show_original = False
	verbose = True
	filenames = []
	maxnum = 10
	evals = 1000
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	for num in xrange(1,maxnum):
		filenames.append('hj{}.dat'.format(num))
	for filename in filenames:
		A,c_x,c_y,tau,m,n,u,x,y,z = lstsq_cubic(filename,ratio,verbose)
		param_eval = np.linspace(start=tau[0],stop=tau[-1],num=evals,endpoint=True)
		x_eval = np.zeros(evals)
		y_eval = np.zeros(evals)
		c_x_eval = refine2(c_x,3)
		c_y_eval = refine2(c_y,3)
		for i,param_i in enumerate(param_eval):
			x_eval[i] = alg220(3,tau,c_x_eval,param_i)
			y_eval[i] = alg220(3,tau,c_y_eval,param_i)
			pass
		print x_eval[0],y_eval[0]
		print x_eval[-1],y_eval[-1]
		ax.plot(x,y,z[0],'+',ms=1)
		ax.plot(c_x,c_y,z[0],'o',ms=5)
		ax.plot(x_eval,y_eval,z[0],label='spline z={}'.format(z[0]))
		#ax.plot(np.dot(A,c_x),np.dot(A,c_y),z[0],label='lstq sol')
	ax.legend(prop={'size':6})
	plt.savefig('{}.png'.format(ratio))

	pass

showcase()







