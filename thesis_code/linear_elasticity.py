#Python FEM FENICS Solver 2016
#Daniel Osen
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

#Test Constant Solution

def constant_solver(N,string_meshtype,element_order):
	
	#Define lame constants for isotropic homogeneous media
	mu = 1.0
	lambda_ = 1.0

	if string_meshtype == "box":
		#Unit box with N-sized partitions in x,y,z
		mesh = BoxMesh(0.0,0.0,0.0,1.0,1.0,1.0,N,N,N)

	elif string_meshtype =="sphere":
		#unit sphere of N? points
		mesh = Mesh(Sphere(Point(0.0,0.0,0.0),1.0),N)


	#Define boundary tolerance (only used for bugtesting)
	tol = 1E-3

	#Define Dirichlet (clamped) boundary
	if string_meshtype == "box":

		def clamped_boundary(x,on_boundary):
			#The unit cube has 0<=x<=1, 0<=y<=1, 0<=z<=1
			#Hence if x,y,z = 0 or 1 we are on the boundary
			#This detection is imprecise, hence we use the near function
			#to determine if the element is close enough to the boundary
			'''
			for x_i in x:
				if near(x_i,1.0) or near(x_i,0.0):
					return True
			return False
			'''

			#fortunately, fenics will do this for us with the supplied
			#on_boundary argument

			#update: we use a single side of the box where
			#y=0 and z=0 as a neumann boundary instead
			#for some reason this fails entirely..
			#we prescribe everywhere a dirichlet boundary then
			return on_boundary

	elif string_meshtype == "sphere":
		def clamped_boundary(x,on_boundary):
			#similarly to the cube case
			#we use the near argument to check if the distance
			#from the origin to the point x is sufficiently close to 1
			'''
			x_r = 0
			for x_i in x:
				x_r +=x_i**2
			x_r = sqrt(x_r)

			if near(x_r,1.0):
				return True
			else:
				return False
			'''
			#once again, fenics already supplies if a point is on the boundary
			#on the mesh, and since we have a full clamped boundary
			#there is no need to do any additional computation
			return on_boundary

	else:
		raise ValueError("Incorrect Mesh!")


	#scaling parameter
	d = 1.0

	#Define function space
	V = VectorFunctionSpace(mesh,element_order,1)
	#Define source term
	#the source term is zero since the solution is a constant (or linear)
	#unless its not..
	f = Constant((-d*lambda_-2*d*mu,0.0,0.0))



	#We must prescribe this constant value as a dirichlet boundary
	#otherwise the system is degenerate
	#we can choose any constant value regardless
	#our choice is the vector constant (1,1,1)
	u_D = Expression(('d*0.5*(1-x[0])*(1-x[0])','0.0','0.0'),degree=2,d=d)
	bc = DirichletBC(V, u_D, clamped_boundary)

	#We now define the weak form of the pde
	#for the approximation u

	#symmetric strain-displacement equation
	def strain(u):
		return 0.5*(nabla_grad(u)+nabla_grad(u).T)

	#the istropic stress-strain relation or stress tensor:
	def stress(u):
		#as far as i know fenics doesnt have a built-int Trace command
		#but its easy to prove the following result:
		trace_strain = nabla_div(u)*Identity(u.geometric_dimension())
		#hence we return for the isotropic media
		return lambda_*trace_strain + 2*mu*strain(u)

	#Define our trial-function
	u = TrialFunction(V)
	v = TestFunction(V)
	#since we do not have a neumann boundary, there is no traction condition
	#and with a constant solution, the traction is zero anyway 
	#(the stress and strain is zero everywhere)
	#we define it anyway for other problems
	if string_meshtype == "box":
		T = Constant((0.0,0.0,0.0))
	elif string_meshtype == "sphere":
		pass

	#We define our linear system to be solved (weak form pde)
	#the bilinear operator
	a = inner(stress(u),strain(v))*dx
	L = dot(f,v)*dx#+dot(T,v)*ds

	#solve system
	u = Function(V)
	solve(a==L,u,bc)
	#plot solution
	#plot(u, title='Displacement', mode='displacement',interactive=True)

	#compute error
	#our exact solution is a constant, equal to the dirichlet boundary condition
	#unless we dont want it too..
	error_L2 = errornorm(u_D,u,'L2',degree_rise=1)
	print(error_L2)
	'''
	
	print(error_L2)
	if string_meshtype == "box":
		print(mesh.num_cells())
	vtkfile = File('constant_solution.pvd')
	vtkfile << u
	'''
	return error_L2,mesh.num_cells()

if __name__ == "__main__":
	E = []
	N = []
	H =[]
	for i in range(0,5):
		n = int(2**i)
		e,h = constant_solver(n,"box","P")
		E.append(e)
		H.append(h)
		N.append(1.0/n)
	print(E)
	print(N)
	#E = np.log(E)
	#N = np.log(N)
	#plt.plot(np.log(N),np.log(E))
	plt.show()
	r = np.log(E[-1]/E[-2])/np.log(N[-1]/N[-2])
	print(r)
	r = np.log(E[-1]/E[-2])/np.log(H[-1]/H[-2])
	print(r)
	r = np.log(H[-1]/H[-2])/np.log(N[-1]/N[-2])
	print(r)
	#Do computation:
	'''
	maxit = 4
	nvals = np.zeros(maxit)
	l_inf = np.zeros(maxit)
	l_2 = np.zeros(maxit)
	for i in range(0,maxit):
		l_inf[i],l_2[i],nvals[i] = my_solver(2**(i+1))

	r = np.zeros(maxit)
	g = np.zeros(maxit)
	for i in range(1,maxit):
		r[i-1] = np.log(l_2[i]/l_2[i-1])/np.log(nvals[i-1]/nvals[i])
		g[i-1] = np.log(l_inf[i]/l_inf[i-1])/np.log(nvals[i-1]/nvals[i])
	print("convergence rate L2 = {}".format(r))
	print("convergence rate Linf = {}".format(g))
	
	'''
