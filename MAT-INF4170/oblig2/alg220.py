import numpy as np
import binarysearch
import matplotlib.pyplot as plt 

def get_knot_averages(t,p,n):
	knot_averages = np.zeros(n)
	for i in range(0,n):
		knot_averages[i] = np.mean(t[(i+1):(i+p+1)])
	return knot_averages


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
	mu = binarysearch.binary_find_mu(x,t)

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
	mu = binarysearch.binary_find_mu(x,t)

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

def showcase(case,plot):
	if case==0:
		t = np.array([0,0,0,0,1,1,2,2,2,4,5,5,5,5],dtype=float)
		p = 3
		c_0 = np.array([0,3,1,4,6,1,5,3,0,4],dtype=float)
	elif case==1:
		t = np.array([0, 0, 0, 1, 1, 2, 3, 3, 3],dtype=float)
		p = 2
		c_0 = np.array([1, 0, 2, 0.5, 0, 1],dtype=float)
	elif case==2:
		t = np.array([-1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 5, 5, 5],dtype=float)
		p = 3
		c_0 = np.array([-1, 1, -1, 1, -1, 1, -1, 1, -1],dtype=float)
	knot_averages = get_knot_averages(t,p,c_0.size)
	x = np.linspace(t[0],t[-1],1000)
	y = np.zeros(shape=x.shape,dtype=float)
	for i,x_i in enumerate(x):
		y[i] = alg220(p,t,c_0,x_i)
	plt.plot(knot_averages,c_0,'-o',x,y)
	plt.legend(('Control Polygon','Spline'))
	plt.xlabel('x')
	plt.ylabel('f(x)')
	if plot:
		plt.show() 
	else:
		plt.savefig('{}.png'.format(case))

showcase(2,plot=True)