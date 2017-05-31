import numpy as np
import matplotlib.pyplot as plt 
from scipy.spatial.distance import cdist

#binary_mu , get_knot_averages and alg220 copied from oblig2

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

def get_knot_averages(t,p,n):
	knot_averages = np.zeros(n)
	for i in range(0,n):
		knot_averages[i] = np.mean(t[(i+1):(i+p+1)])
	return knot_averages

#oblig_3 code

def alg220_modified(mu,p,t,c_0,x,i):
	#assumes we have enough knots, t here is actually tau and x is t

	if mu == -1:
		'''x not in any t_mu <= x < t_{mu+1}'''
		print('endpoint reached')
		return c_0[-1]

	c = np.copy(c_0)
	m = 0
	for k in range(p,0,-1):
		for j in range(mu,mu-k+1-1,-1):
			c[j-m] = np.divide(t[j+k]-x[i+k],t[j+k]-t[j])*c[j-1-m] + np.divide(x[i+k]-t[j],t[j+k]-t[j])*c[j-m]
		c = c[1:]
		m+=1
	return c[mu-m]

def get_b(p,tau,t,m,c):
	#oslo algorithm

	b = np.zeros(m)

	#calculate b
	for i in range(0,m):

		mu = binary_find_mu(t[i],tau)

		if p==0:

			b[i] = c[mu-p]

		elif p>0:

			b[i] = alg220_modified(mu,p,tau,c,t,i) 

	return b


def showcase(plotme=False):

	#define Example 4.7, corresponding to figure 4.5
	p = 2
	tau = np.array([-1,-1,-1,0,1,1,1],dtype=float)
	c = np.array([1,-2,2,-1],dtype=float)

	#calculate knot averages for unrefined control polygon
	average_original = get_knot_averages(tau,p,c.size)

	#calculate new coefficients for the first refinement
	t_1 = np.array([-1,-1,-1,-0.5,0,0.5,1,1,1],dtype=float)
	m_1 = t_1.size-(p+1)
	b_1 = get_b(p,tau,t_1,m_1,c)

	#check that implementation returns answer from Example 4.7
	b_correct = np.array([1,-0.5,-1,1,0.5,-1],dtype=float)
	assert np.allclose(b_1,b_correct)

	#calulate knot averages for first refined control polygon
	average_refined_1 = get_knot_averages(t_1,p,b_1.size)

	#calculate new coefficients for the second refinement
	t_2 = np.array([-1,-1,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1,1],dtype=float)
	m_2 = t_2.size-(p+1)
	b_2 = get_b(p,tau,t_2,m_2,c)

	#calculate knot averages for second refined control polygon
	average_refined_2 = get_knot_averages(t_2,p,b_2.size)

	#calculate new coefficients for the third and last refinement
	t_3 = np.array([-1,-1,-1,-0.875,-0.75,-0.625,-0.5,
		-0.375,-0.25,-0.125,0,0.125,0.25,0.375,0.5,0.625,0.75,0.875,1,1,1])
	m_3 = t_3.size - (p+1)
	b_3 = get_b(p,tau,t_3,m_3,c)

	#calculate knot averages for the third and last refined control polygon
	average_refined_3 = get_knot_averages(t_3,p,b_3.size)

	#prepare for plotting
	x = np.linspace(tau[0],tau[-1],1000)
	y = np.zeros(shape=x.shape,dtype=float)

	#evaluate the original spline
	for i,x_i in enumerate(x):
		y[i] = alg220(p,tau,c,x_i)

	#plot the results
	plt.plot(average_original,c,'--',average_refined_1,b_1,'--',
		average_refined_2,b_2,'--',average_refined_3,b_3,'--',x,y)
	plt.legend(('m=4','m={}'.format(m_1),'m={}'.format(m_2),'m={}'.format(m_3),'spline'))
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Converging Control Polygon')
	if plotme:
		plt.show()
	else:
		plt.savefig('plot.png')

	#Calculate the numerical hausdorff distance using euclidean metric
	#and linear interpolation on the control polygons

	#get linear interpolants
	p_0 = np.interp(x,average_original,c).reshape(x.size,1)
	p_1 = np.interp(x,average_refined_1,b_1).reshape(x.size,1)
	p_2 = np.interp(x,average_refined_2,b_2).reshape(x.size,1)
	p_3 = np.interp(x,average_refined_3,b_3).reshape(x.size,1)
	x = x.reshape(x.size,1)
	y = y.reshape(x.size,1)

	#get numerical euclidean distance to spline curve
	d_0 = cdist(np.hstack((x,y)),np.hstack((x,p_0)),'euclidean')
	d_1 = cdist(np.hstack((x,y)),np.hstack((x,p_1)),'euclidean')
	d_2 = cdist(np.hstack((x,y)),np.hstack((x,p_2)),'euclidean')
	d_3 = cdist(np.hstack((x,y)),np.hstack((x,p_3)),'euclidean')

	#get hausdorf distance max(sup_x inf_y d(x,y), sup_y inf_x d(x,y))
	h1 = np.max(np.min(d_0,axis=1))
	h2 = np.max(np.min(d_0,axis=0))
	dist = max(h1,h2)
	print "Hausdorff Distance = {}".format(dist)
	h1 = np.max(np.min(d_1,axis=1))
	h2 = np.max(np.min(d_1,axis=0))
	dist = max(h1,h2)
	print "Hausdorff Distance = {}".format(dist)
	h1 = np.max(np.min(d_2,axis=1))
	h2 = np.max(np.min(d_2,axis=0))
	dist = max(h1,h2)
	print "Hausdorff Distance = {}".format(dist)
	h1 = np.max(np.min(d_3,axis=1))
	h2 = np.max(np.min(d_3,axis=0))
	dist = max(h1,h2)
	print "Hausdorff Distance = {}".format(dist)



showcase()




