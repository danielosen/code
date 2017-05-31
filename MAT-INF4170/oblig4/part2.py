import numpy as np
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import csv
from matplotlib import cm

def find_mu_oblig2(x,t,mu):
	''' I have copied the find mu algorithm from oblig2 since my binary search one has a bug in it...'''
	
	if x<t[mu]:
		
		mu = 0
	
	while t[mu+1] <= x and t[mu+1]<t[-1]:
		
		mu = mu+1
	
	return mu

def alg220(p,t,c_0,x,mu):
	''' Changed my own implementation of alg220 to account for the new find_mu algorithm'''
	
	mu = find_mu_oblig2(x,t,mu)

	c = np.copy(c_0)
	
	m = 0
	
	for k in range(p,0,-1):
	
		for j in range(mu,mu-k+1-1,-1):
	
			c[j-m] = np.divide(t[j+k]-x,t[j+k]-t[j])*c[j-1-m] + np.divide(x-t[j],t[j+k]-t[j])*c[j-m]
	
		c = c[1:]
	
		m+=1
	
	return c[mu-m]

def read_data(filename):
	'''reads a .dat file and outputs arrays of x,y,z data'''

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
	'''Computes chord length parametrization and knot vector tau'''
	
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

def get_filenames(maxnum):
	'''generates the first maxnum-1 filenames'''
	
	filenames = []
	
	for num in xrange(1,maxnum):
	
		filenames.append('hj{}.dat'.format(num))
	
	return filenames

def get_lstsq_cubic_v3(filename,ratio):
	'''Construct the least squares system and solve it'''

	p=3
	
	x,y,z = read_data(filename)
	
	u,tau,n,m = process_data(x,y,ratio)
	
	A = np.zeros((m,n),dtype=float)
	
	for i in xrange(m):
	
		for j in xrange(n):
	
			A[i,j] = cubic_bspline_eval(tau,u[i],j)
	
	N = np.dot(A.T,A)
	
	#removed rank calculation after making the error plot
	'''
	if np.linalg.matrix_rank(N)< n:
		
		print('SINGULAR MATRIX')
		
		c_x,_,_,_ = np.linalg.lstsq(A,x)
		
		c_y,_,_,_ = np.linalg.lstsq(A,y)
		else:
	'''

	c_x = np.linalg.solve(N,np.dot(A.T,x))
		
	c_y = np.linalg.solve(N,np.dot(A.T,y))
	
	return A,c_x,c_y,z,tau,x,y

def cubic_bspline_eval(tau,u_i,j):
	'''Evaluate B_{j,p=3,tau}(u_i)'''
	
	p = 3
	
	tau_aug = np.copy(tau[j:j+p+1+1])
	
	left = np.zeros(p+1,dtype=float)+tau_aug[0]-100*np.abs(tau_aug[1]+tau_aug[0])
	
	right = np.zeros(p+1,dtype=float)+tau_aug[-1]+100*np.abs(tau_aug[-1]-tau_aug[-2])
	
	tau_aug = np.concatenate((left,tau_aug))
	
	tau_aug = np.concatenate((tau_aug,right))
	
	c = np.zeros(1+2*(p+1))
	
	c[0+p+1]= 1.0
	
	out = alg220(p,tau_aug,c,u_i,0)
	
	return out

def cubic_spline_eval(tau,u,c):
	'''Evaluate \sum_j B_{j,p=3,tau}(u)'''

	p = 3

	tau_aug = np.copy(tau)

	left = np.zeros(p+1,dtype=float)+tau_aug[0]-1000*np.abs(tau_aug[1]+tau_aug[0])

	right = np.zeros(p+1,dtype=float)+tau_aug[-1]+1000*np.abs(tau_aug[-1]-tau_aug[-2])

	tau_aug = np.concatenate((left,tau_aug))

	tau_aug = np.concatenate((tau_aug,right))

	c_aug = np.concatenate((np.zeros(p+1,dtype=float),c))

	c_aug = np.concatenate((c_aug,np.zeros(p+1,dtype=float)))

	out = np.zeros(u.size)

	for i in range(u.size):

		out[i] = alg220(p,tau_aug,c_aug,u[i],0)

	return out

def generate_plot(ratio):
	''' generates the curve plots for problem 1 with n=ratio*m'''

	filenames = get_filenames(10)

	fig = plt.figure()

	ax = fig.gca(projection='3d')

	for filename in filenames:

		A,c_x,c_y,z,tau,data_x,data_y = get_lstsq_cubic_v3(filename,ratio)	

		ax.plot(c_x,c_y,z[0],'o')

		ax.plot(data_x,data_y,z[0],'+')
		
		ax.plot(np.dot(A,c_x),np.dot(A,c_y),z[0],label='z = {}'.format(z[0]))
	
	ax.legend(prop={'size':6})
	
	plt.savefig('{}.png'.format(ratio))	

def generate_error_plot():
	''' generates the error plot for problem 1'''

	ratios = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9],dtype=float)

	error_tot = np.zeros(ratios.size)

	for i in range(ratios.size):

		filenames = get_filenames(10)

		error_x = np.zeros(9)
		
		error_y = np.zeros(9)

		for j in range(9):

			A,c_x,c_y,z,tau,data_x,data_y = get_lstsq_cubic_v3(filenames[j],ratios[i])

			error_x[j] = np.linalg.norm(np.dot(A,c_x)-data_x)

			error_y[j] = np.linalg.norm(np.dot(A,c_y)-data_y)

		error_tot[i] = np.sqrt( np.mean(error_x)**2 +np.mean(error_y)**2)

	plt.plot(ratios,error_tot)

	plt.savefig('errorplot.png')



def sample_splines(ratio):
	'''Samples 20 points from each spline curve solution to the least squares problem'''

	filenames = get_filenames(10)
	
	j = 0

	data = {}

	for filename in filenames:

		A,c_x,c_y,z,tau,data_x,data_y = get_lstsq_cubic_v3(filename,ratio)

		u = np.linspace(start=tau[0],stop=tau[-1],num=20,endpoint=True,dtype=float)

		x = cubic_spline_eval(tau,u,c_x)

		y = cubic_spline_eval(tau,u,c_y)

		data[j] = {'x': x, 'y': y, 'z': z}
		
		j+=1

	return data

def get_surface_knots():

	u = np.linspace(start=0,stop=1.0,num=20,endpoint=True,dtype=float)

	tau_u = np.zeros(15,dtype=float)

	tau_u[0:4] = u[0]

	tau_u[15-4:15] = u[-1]

	tau_u[4:15-4] = np.linspace(start=u[1],stop=u[-2],num=7,endpoint=True,dtype=float)
	
	v = np.linspace(start=0,stop=1.0,num=9,endpoint=True,dtype=float)

	tau_v = np.zeros(9,dtype=float)

	tau_v[0:4] = v[0]

	tau_v[9-4:9] = v[-1]

	tau_v[4] = 0.5*(v[0]+v[-1])

	n1 = 11

	m1 = 20

	n2 = 5

	m2 = 9

	return u,tau_u,v,tau_v,m1,n1,m2,n2

def cubic_surface_lstsq(ratio):

	data = sample_splines(ratio)

	u,tau_u,v,tau_v,m1,n1,m2,n2 = get_surface_knots()

	#build matrix A_{i,j} = \phi_{j,3,\tau_u} (u_i)

	A = np.zeros((m1,n1))

	for i in range(m1):

		for j in range(n1):

			A[i,j] = cubic_bspline_eval(tau_u,u[i],j)

	#build matrix B_{i,j} = \psi_{j,3,\tau_v} (v_i)

	B = np.zeros((m2,n2))

	for i in range(m2):

		for j in range(n2):

			B[i,j] = cubic_bspline_eval(tau_v,v[i],j)

	#build the F matrices

	F_x = np.zeros((m1,m2))

	F_y = np.zeros((m1,m2))
	
	F_z = np.zeros((m1,m2))

	for i in range(m1):

		for j in range(m2):

			F_x[i,j] = data[j]['x'][i]

			F_y[i,j] = data[j]['y'][i]

			F_z[i,j] = data[j]['z'][0] #const

	#Solve for the cofficients (c_x)_{i,j} (c_y)_{i,j}, (c_z)_{i,j}

	N_1 = np.dot(A.T,A)

	N_2 = np.dot(B.T,B)

	D = np.linalg.solve(N_1,np.dot(A.T,F_x))

	C_x = np.linalg.solve(N_2,np.dot(B.T,D.T))

	C_x = C_x.T

	D = np.linalg.solve(N_1,np.dot(A.T,F_y))

	C_y = np.linalg.solve(N_2,np.dot(B.T,D.T))

	C_y = C_y.T

	D = np.linalg.solve(N_1,np.dot(A.T,F_z))

	C_z = np.linalg.solve(N_2,np.dot(B.T,D.T))

	C_z = C_z.T

	#plot coefficients

	fig = plt.figure()

	ax = fig.gca(projection='3d')

	#plot ctrlpts 

	'''
	for j in range(n2):

		ax.plot(C_x[:,j],C_y[:,j],C_z[:,j],'o') #ctrl pts
	'''

	#plot lstsq solution
	'''

	X_lstsq = np.dot(A,C_x)

	X_lstsq = np.dot(X_lstsq,B.T)

	Y_lstsq = np.dot(A,C_y)

	Y_lstsq = np.dot(Y_lstsq,B.T)

	Z_lstsq = np.dot(A,C_z)

	Z_lstsq = np.dot(Z_lstsq,B.T)

	#ax.plot_surface(X_lstsq,Y_lstsq,Z_lstsq) #bugged? looks very strange, faces are only visible 'inside'

	#trying to plot curves by fixing i and j each, looks much better, but no filling in between patches

	#still, it looks very coarse

	for j in range(m2):
		
		ax.plot(X_lstsq[:,j],Y_lstsq[:,j],Z_lstsq[:,j])

	for i in range(m1):

		ax.plot(X_lstsq[i,:],Y_lstsq[i,:],Z_lstsq[i,:])

	'''

	#try to refine evaluations and plot again, this looks nice !

	u_eval = np.linspace(start=u[0],stop=u[-1],num=100,endpoint=True,dtype=float)
	
	v_eval = np.linspace(start=v[0],stop=v[-1],num=100,endpoint=True,dtype=float)

	X = np.zeros((100,100))

	Y = np.zeros((100,100))

	Z = np.zeros((100,100))

	for l in range(100):
			
			for j in range(n2):

				X[:,l] += cubic_spline_eval(tau_u,u_eval,C_x[:,j])*cubic_bspline_eval(tau_v,v_eval[l],j)

				Y[:,l] += cubic_spline_eval(tau_u,u_eval,C_y[:,j])*cubic_bspline_eval(tau_v,v_eval[l],j)

				Z[:,l] += cubic_spline_eval(tau_u,u_eval,C_z[:,j])*cubic_bspline_eval(tau_v,v_eval[l],j)

	ax.plot_surface(X,Y,Z)


	#plt.savefig('surface.png')
	plt.show()
	




def plot_sampled_data():

	ratio = 0.1

	data = sample_splines(ratio)

	fig = plt.figure()

	ax = fig.gca(projection='3d')

	for entry in data:

		ax.plot(data[entry]['x'],data[entry]['y'],data[entry]['z'][0],'o',label='z={}'.format(data[entry]['z']) )

	#plt.savefig('sampled_points.png')

	plt.show()



#to plot ratio=0.05, 0.1, 0.2 etc.
cubic_surface_lstsq(0.1)