# Exercise 1

import numpy as np
from numpy import sin,cos
import matplotlib.pyplot as plt 

pi = 3.1415926535897932

def solver(N,plot=False):
	'''solve variational problem on uniform grid illustrating gibbs phenomenon'''
	M = 10000
	u = np.zeros(M)
	x = np.linspace(0,1.0,num=M)

	for i in range(1,M+1):
		for j in range(1,N+1):
			u_k = 2*(( cos(pi*j/3)-cos(2*pi*j/3))/(pi*j))
			u[i-1] += u_k*sin(pi*j*x[i-1])

	if plot:
		plt.plot(x,u)
		plt.xlabel("x")
		plt.ylabel("u(x)")
		plt.title("Gibbs Phenomenon")
		plt.show()
	else:
		return(u,u_k,x)

def error(N,norm):
	'''compute the error for a given numerical norm'''

	u,u_k,x = solver(N)
	error = 0
	dx = x[1]-x[0]
	xmin = 1/float(3)
	xmax = 2/float(3)

	if norm=="L2":
		for i in range(1,N+1):
			if x[i-1] >= xmin and x[i-1] <= xmax:
				error += np.abs(u[i-1]-1)**2*dx
			else:
				error += np.abs(u[i-1])**2*dx
		error = sqrt(error)
	elif norm=="Linf":
		for i in range(1,N+1):
			if x[i-1] >= xmin and x[i-1] <= xmax:
				error = max( error , np.abs(u[i-1]-1) )
			else:
				error = max( error , np.abs(u[i-1]) )
		pass
	elif norm=="H1":
		#calculate value of derivates by using basis functions
		#dirac's delta (step function derivative) is not in l2 so its infinite
		return -1

		pass

	return(error)

def showcase_error(norm):
	'''the overshoot in gibbs phenomenon approaches a finite limit, hence does the error'''
	N_list = [10,20,30,40,50,60,70,80,90,100,500]
	E_list = []
	for N in N_list:
		E_list.append(error(N,norm))
	print(E_list)
	plt.plot(N_list,E_list)
	plt.show()

def showcase_solution():
	N_list = [10,20,30,40,50,60,70,80,90,100,500]
	E_list = []
	for N in N_list:
		solver(N,True)

if __name__ == '__main__':
	showcase_solution()

	

