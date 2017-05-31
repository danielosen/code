#MEK4250 Oblig 1

#IMPORTS
from dolfin import *
import numpy as np

#DEFS 
def my_solver(n,mu_,element_family,degree,plotme=False):

	def boundary(x,on_boundary):
		if on_boundary:
	#		if near(x[0],0.0) or near(x[0],1.0):
			return True
		return False

	u_dirichlet_boundary = Expression('near(x[0],0.0) ? 0.0 : 1.0') #since the boundary returns true only if we are near 0 or 1, this works.

	mu = Constant(mu_)
	

	#Generate Mesh (0,1)^2
	mesh = UnitSquareMesh(n,n)

	beta = Constant(0.5*mesh.hmin())

	#Define Variational Problem
	V = FunctionSpace(mesh,element_family,degree)
	u = TrialFunction(V)
	v = TestFunction(V)
	u_e = Expression('(exp(1.0/mu*x[0])-1.0)/(exp(1/mu)-1.0)',mu=mu)
	a = mu*dot(grad(u),grad(v))*dx+grad(u)[0]*v*dx + beta*grad(u)[0]*grad(v)[0]*dx
	if degree >= 2:
		#a -= mu*beta*div(grad(u))*grad(v)[0]*dx
		a_1 = Constant(0)*a +mu*beta*dot(grad(u),grad(grad(v)[0]))*dx
		a_2 = Constant(0)*a - mu*beta*div(grad(u))*grad(v)[0]*dx
	L = Constant(0.0)*(v+grad(v)[0])*dx

	#Define dirichlet boundary condition
	bc = DirichletBC(V,u_dirichlet_boundary,boundary)

	#Compute Solution
	u = Function(V)

	#check something
	#A_1 = assemble(mu*beta*dot(grad(u),grad(grad(v)[0]))*dx)
	#A_2 = assemble(-mu*beta*div(grad(u))*grad(v)[0]*dx)
	#print allclose(A_1.array(),A_2.array())
	A1 = assemble(a_1)
	A2 = assemble(a_2)
	print A1.array()-A2.array()
	bc.apply(A1)
	bc.apply(A2)
	print np.max(np.max(A1.array()-A2.array()))
	print (A1.array()-A2.array())


	solve(a==L,u,bc)

	#Plot Solution
	if plotme:
		plot(mesh,interactive=True)
		plot(u,interactive=True)

	#Compute Error
	print("L2 Error: {:.2E}".format(errornorm(u_e,u,'L2')))
	print("H1 Error: {:.2E}".format(errornorm(u_e,u,'H1')))


if __name__ == "__main__":
	mu = 0.1
	my_solver(8,mu,'CG',2)
	#my_solver(16,mu,'CG',2)
	#my_solver(32,mu,'CG',2)



