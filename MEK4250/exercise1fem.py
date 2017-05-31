#exercise1fem

import numpy as np
import matplotlib.pyplot as plt
from dolfin import *

def my_solver(N,normname):
	mesh = IntervalMesh(N,0,1)
	f = Expression('x[0]<=2/3.0 && x[0] >= 1/3.0')
	V = FunctionSpace(mesh,'P',1)
	u = TrialFunction(V)
	v = TestFunction(V)
	L = inner(f,v)*dx
	a = inner(u,v)*dx
	u_N = Function(V)
	solve(a == L,u_N)
	#L2, H1, linf
	if normname == 'Linf':
		ff = interpolate(f,V)
		error = np.abs(ff.vector().array() - u_N.vector().array()).max()
	else:
		error = errornorm(f,u_N,norm_type=normname,degree_rise=3,mesh=mesh)
	
	#print("Error {}: {}".format(normname,error))
	plot(u_N,interactive=True)
	return error


if __name__ == '__main__':
	N = [3,9,18,27,100]
	norms = ['H1','L2','Linf']
	errors = {}
	errors['H1'] = []
	errors['L2'] = []
	errors['Linf'] = []
	for n in N:
		for norm in norms:
			error = my_solver(n,norm)
			errors[norm].append(error)
	for norm in norms:
		plt.plot(N,errors[norm])
		plt.title(norm)
		plt.show()


