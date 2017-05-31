import numpy as np
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
	tau = np.linspace(start=u[0],stop=u[-1],num=n+p+1-2*(p+1)+2,endpoint=True,dtype=float)
	tau = np.concatenate((np.zeros(3)+tau[0],tau))
	tau = np.concatenate((tau,np.zeros(3)+tau[-1]))
	return u,tau,n,m

def cubic_bspline_eval(j,u_i,tau):
	'''evaluates the cubic B-spline at u_i: B_{3,j,tau_j,...tau_j+p+1}(u_i)
	
	c, memory and overwriting layout at each step, b_j,p is the j'th b_spline of order p

	  l=0      l=1       l=2       l=3
	[ b_j_0 | b_j+1_0 | b_j+2_0 | b_j+3_0 ] k=0
	[ x     | b_j,1   | b_j+1,1 | b_j+2,1 ] k=1
	[ x     | x       | b_j,2   | b_j+1,2 ] k=2
	[ x     | x       | x       | b_j,3   ] k=3

	I think there is a bug here? Found it!
	'''
	c = np.zeros(4,dtype=float)
	#zero-order b-splines
	for l in xrange(3,-1,-1): #3,2,1,0
		if tau[j+l] <= u_i and u_i < tau[j+l+1]: #tau_[j+4],...,tau_[j] 
			c[l] = 1.0
			break
	#remaining b-splines
	for k in range(1,4):
		#1,2,3
		for l in range(3,-1+k,-1): 
			#3,2,1 - 3,2 - 3
			#We enforce the zero-rule in this implementation, near machine precision at least, so we don't have to worry about the knot vector too much.
			if np.abs(tau[j+l]-tau[j+(l-k)])<10**-16:
				left = 0.0
			else:
				left = np.divide(u_i - tau[j+(l-k)],tau[j+l]-tau[j+(l-k)]) * c[l-1]

			if np.abs(tau[j+(l+1)]-tau[j+(l-k+1)])<10**-16:
				right = 0.0
			else:
				right = np.divide(tau[j+(l+1)]-u_i,tau[j+(l+1)]-tau[j+(l-k+1)])

			c[l] = left * c[l-1]   +   right * c[l]
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
			A[i,j] = np.asscalar(cubic_bspline_eval2(j,u[i],tau,n))
	N = np.dot(A.T,A)
	rank = np.linalg.matrix_rank(N)
	if rank<n:
		print ('WARNING: Singular Matrix!')
		c_x,_,_,_ = np.linalg.lstsq(A,x)
		c_y,_,_,_ = np.linalg.lstsq(A,y)
	else:
		c_x = np.linalg.solve(N,np.dot(A.T,x))
		c_y = np.linalg.solve(N,np.dot(A.T,y))
	return A,c_x,c_y,tau,m,n,u,x,y,z

def get_lstsq_cubic(filename,ratio):
	'''for use in Problem 2'''


	x,y,z = read_data(filename)
	u,tau,n,m = process_data(x,y,ratio)
	A = np.zeros((m,n),dtype=float)
	for i in xrange(m):
		for j in xrange(n):
			A[i,j] = np.asscalar(cubic_bspline_eval2(j,u[i],tau,n))
	N = np.dot(A.T,A)
	c_x = np.linalg.solve(N,np.dot(A.T,x))
	c_y = np.linalg.solve(N,np.dot(A.T,y))
	return c_x,c_y,z,tau,x,y

def refine2(c,p):
	c_ref = np.copy(c)
	left = np.zeros(p+1)
	right = np.zeros(p+1)+c[0]
	c_ref = np.concatenate((left,c))
	c_ref = np.concatenate((c,right))
	return c_ref

def get_filenames(maxnum):
	filenames = []
	for num in xrange(1,maxnum):
		filenames.append('hj{}.dat'.format(num))
	return filenames

def sample_spline_curves(ratio,verbose):
	maxnum = 10
	filenames = get_filenames()
	for filename in filenames:
		A,c_x,c_y,tau,m,n,u,x,y,z = lstsq_cubic(filename,ratio,verbose)
		U_j = u[-1]


def test_cubic_bspline_eval(tau):
	'''tests my implementation on special values'''
	u_eval = np.linspace(start=tau[0],stop=tau[-1],num=100,endpoint=True,dtype=float)
	out = np.zeros(100)
	for i in range(100):
		out[i] = cubic_bspline_eval2(j=0,u_i=u_eval[i],tau=tau,n=10)
	print "knot vector for cubic B-spline: {}".format(tau)
	plt.plot(u_eval,out)
	plt.show()

def cubic_bspline_eval2(j,u_i,tau,n):

	#cubic
	p=3
	#grab support of the j'th cubic B-spline
	tau_support = np.copy(tau[j:j+p+1+1])
	if j== n-1:
		#this guy has to do with connecting the last point...? Get garbage otherwise
		return np.array([1.0])
	#augment this support with a,a,a,a, .... , b,b,b,b
	#where a<tau_support[0] and b>tau_support[-1]
	left = np.zeros(p+1,dtype=float) + tau_support[0]-np.abs(tau_support[1]-tau_support[0])
	right = np.zeros(p+1,dtype=float) + tau_support[-1]+np.abs(tau_support[-1]-tau_support[-2])
	tau_augmented = np.concatenate((left,tau_support))
	tau_augmented = np.concatenate((tau_augmented,right))
	#solve for the coefficients in the larger space
	#To recover the exact same B-spline, we merely zero out all the coefficients belonging to 
	#B-splines which depends on the new knots
	#that leaves only 1 B-spline left, corresponding to index j+p+1, and this guy depends
	#on exactly the same knots, so therefore the coefficient must be 1
	c = np.zeros(1+2*(p+1))
	c[0+p+1] = 1.0
	out = alg220(p,tau_augmented,c,u_i)
	return out

'''
def refine(v,p,knot):
	left = np.zeros(p+1,dtype=float)
	right = np.zeros(p+1,dtype=float)
	if knot:
		left += v[0]-np.abs(v[1]-v[0])
		right += v[-1]+np.abs(v[-1]-v[-2])
	out = np.concatenate((left,v))
	out = np.concatenate((v,right))
	return out
'''







def showcase(ratio):
	'''show least squares approximations'''
	verbose = False
	filenames = get_filenames(10)
	evals = 1000
	errors_x = np.zeros(9)
	errors_y = np.zeros(9)
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	e = 0
	for filename in filenames:
		A,c_x,c_y,tau,m,n,u,x,y,z = lstsq_cubic(filename,ratio,verbose)
		errors_x[e] = (np.linalg.norm(np.dot(A,c_x)-x))
		errors_y[e] =(np.linalg.norm(np.dot(A,c_y)-y))
		#param_eval = np.linspace(start=tau[0],stop=tau[-1],num=evals,endpoint=True)
		#x_eval = np.zeros(evals)
		#y_eval = np.zeros(evals)
		#c_x_eval = refine(c_x,3,False)
		#c_y_eval = refine(c_y,3,False)
		#tau_eval = refine(tau,3,True)
		#for i,param_i in enumerate(param_eval):
		#	x_eval[i] = alg220(3,tau_eval,c_x_eval,param_i)
		#	y_eval[i] = alg220(3,tau_eval,c_y_eval,param_i)
		ax.plot(x,y,z[0],'+',ms=1)
		#ax.plot(c_x,c_y,z[0],'o',ms=5)
		#ax.plot(x_eval,y_eval,z[0],label='spline z={}'.format(z[0]))
		ax.plot(np.dot(A,c_x),np.dot(A,c_y),z[0],label='z= {}'.format(z[0]))
		e+=1
		#print np.dot(A,c_x)
	ax.legend(prop={'size':6})
	plt.savefig('{}.png'.format(ratio))
	return np.mean(errors_x),np.mean(errors_y)

#test_cubic_bspline_eval(np.array([1.0,2.0,3.0,4.0,5.0],dtype=float))
#test_cubic_bspline_eval(np.array([1.0,1.0,1.0,1.0,5.0],dtype=float))
#test_cubic_bspline_eval(np.array([1.0,3.0,3.0,3.0,5.0],dtype=float))
#test_cubic_bspline_eval(np.array([1.0,5.0,5.0,5.0,5.0],dtype=float))
if __name__ == '__main__':
	ratios = np.array([0.05,0.10,0.20,0.5,0.75,0.9,1.0],dtype=float)
	errors_x = np.zeros(ratios.size)
	errors_y = np.zeros(ratios.size)
	for i in range(ratios.size):
		errors_x[i],errors_y[i] = showcase(ratios[i])
	errors = np.sqrt(errors_x**2.0 + errors_y**2.0)
	fig = plt.figure()
	ax = fig.gca()
	ax.plot(ratios,errors,'+')
	#print np.polyfit(np.log(ratios),np.log(errors),1)
	plt.savefig('error_ratio.png')







