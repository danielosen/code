#algorithm 1.1, Exercise 1.3
import numpy as np
import matplotlib.pyplot as plt

class PolyTree(object):
	'''class i made to store the polynomial q_jk neville-aitken function values, 
	basically indexing into upper half of (p,p) matrix'''
	def __init__(self,p,c):
		self.p = p
		self.array = np.zeros((p+1)*(p+2)/2)
		self.array[0:(p+1)] = c

	def get(self,j,k):
		index=j+(self.p+2)*k - k*(k+1)/2
		return self.array[index]

	def put(self,j,k,value):
		index=j+(self.p+2)*k - k*(k+1)/2
		self.array[index] = value

	def getlast(self):
		return self.array[-1]

class BezierTree(object):
	'''class that stores 1d bezier curve evaluations,
		linear index into lower half of (p,p) matrix'''
	def __init__(self,p,c):
		self.p = p
		self.array = np.zeros((p+1)*(p+2)/2)
		self.array[0:(p+1)] = c

	def get(self,j,k):
		index = j+self.p*k-k*(k-1)/2
		return self.array[index]

	def put(self,j,k,value):
		index = j+self.p*k-k*(k-1)/2
		self.array[index] = value

	def getlast(self):
		return self.array[-1]


def plot_data(xn,yn,xe,ye,x,y,p,error,xlabel,ylabel,title):
	'''plots numerical and analytical results'''
	plt.plot(xn,yn,label="Numerical p={}".format(p))
	plt.plot(xe,ye,label="Analytical l2-error={0:.2f}".format(error))
	plt.plot(x,y,'o',label="Control Points")
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.legend()
	plt.tight_layout()	
	plt.savefig('{}-p{}.png'.format(title,p))
	plt.show()

def uniform_semicircle_sampling(p):
	'''returns p+1 uniformly sampled points (x,y) on the positive semicircle of radius 1'''
	angle_delta = np.divide(np.pi,p)
	x = np.cos(angle_delta * np.arange(p+1))
	y = np.sin(angle_delta * np.arange(p+1))
	return x,y

def parameter_sampling(p,samplemethod):
	'''returns strictly increasing p+1 parameter values sampled uniformly on 0,1 or with random pertubations'''
	if samplemethod == 'uniform':
		return np.linspace(0,1,p+1)
	elif samplemethod == 'random':
		t = np.linspace(0,1,p+1)
		t += np.divide(t[1]-t[0],2)*np.random.ranf((p+1,))
		t = np.divide(t,np.max(np.abs(t)))
		return t
	elif samplemethod == 'asymmetric':
		t1 = np.linspace(0,0.4,(p+1)/2)
		t2 = np.linspace(0.41,1.0,(p+1)/2+(p+1)%2)
		t = np.hstack((t1,t2))
		return t


def neville_aitken(n,p,samplemethod):
	'''Uses the Neville-Aitken Algorithm to compute affine combinations of points x,y,
	building a interpolating polynomial function of degree p, evaluated at
	n+1 points, where parameters are sampled according to samplemethod,
	and returning the evaluations.'''
	x,y = uniform_semicircle_sampling(p)
	t = parameter_sampling(p,samplemethod)
	tn = parameter_sampling(n,'uniform')
	qx = PolyTree(p,x)
	qy = PolyTree(p,y)
	xn = np.zeros(n+1)
	yn = np.zeros(n+1)

	for i,t_i in enumerate(tn):
		for k in range(1,p+1):
			for j in range(0,p-k+1):
				value_x = np.divide(t[j+k]-t_i,t[j+k]-t[j])*qx.get(j,k-1)+ np.divide(t_i-t[j],t[j+k]-t[j])*qx.get(j+1,k-1)
				value_y = np.divide(t[j+k]-t_i,t[j+k]-t[j])*qy.get(j,k-1) + np.divide(t_i-t[j],t[j+k]-t[j])*qy.get(j+1,k-1)
				qx.put(j,k,value_x)
				qy.put(j,k,value_y)
		xn[i] = qx.getlast()
		yn[i] = qy.getlast()

	return xn,yn,x,y,p,n

def l2_error(xn,yn,xe,ye):
	'''returns the l2 norm error'''
	return np.sqrt(np.dot(xn-xe,xn-xe)+np.dot(yn-ye,yn-ye))

def bezier_algorithm(n,p):
	'''implements algorithm 1.2 on same problem as neville-aitken, reusing much of the code'''
	x,y = uniform_semicircle_sampling(p)
	t = parameter_sampling(n,'uniform')
	qx = BezierTree(p,x)
	qy = BezierTree(p,y)
	xn = np.zeros(n+1)
	yn = np.zeros(n+1)

	for i,t_i in enumerate(t):
		for k in range(1,p+1):
			for j in range(k,p+1):
				value_x = (1-t_i)*qx.get(j-1,k-1) + t_i*qx.get(j,k-1)
				value_y = (1-t_i)*qy.get(j-1,k-1) + t_i*qy.get(j,k-1)
				qx.put(j,k,value_x)
				qy.put(j,k,value_y)
		xn[i] = qx.getlast()
		yn[i] = qy.getlast()

	return xn,yn,x,y,p,n






if __name__ == '__main__':
	'''
	#numerical result
	xn,yn,x,y,p,n, = neville_aitken(300,4,'asymmetric')
	#exact result
	xe,ye = uniform_semicircle_sampling(n)
	#error
	error = l2_error(xn,yn,xe,ye)
	plot_data(xn,yn,xe,ye,x,y,p,error,'x(t)','y(t)','Asymmetrically Sampled Parameters')
	'''
	xn,yn,x,y,p,n = bezier_algorithm(300,14)
	xe,ye = uniform_semicircle_sampling(n)
	error = l2_error(xn,yn,xe,ye)
	plot_data(xn,yn,xe,ye,x,y,p,error,'x(t)','y(t)','Bezier Curve')
