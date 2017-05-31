#MEK4250 Oblig 1

#IMPORTS
from dolfin import *


#DEFS 
def my_solver(n,mu,element_family,degree,plotme=False):

	def boundary(x,on_boundary):
		if on_boundary:
			if near(x[0],0.0) or near(x[0],1.0):
				return True
		return False

	u_dirichlet_boundary = Expression('near(x[0],0.0) ? 0.0 : 1.0') #since the boundary returns true only if we are near 0 or 1, this works.

	#Generate Mesh (0,1)^2
	mesh = UnitSquareMesh(n,n)

	#Define Variational Problem
	V = FunctionSpace(mesh,element_family,degree)
	u = TrialFunction(V)
	v = TestFunction(V)
	u_e = Expression('(exp(1.0/mu*x[0])-1.0)/(exp(1/mu)-1.0)',mu=mu)
	a = mu*dot(grad(u),grad(v))*dx+grad(u)[0]*v*dx
	L = Constant(0.0)*v*dx

	#Define dirichlet boundary condition
	bc = DirichletBC(V,u_dirichlet_boundary,boundary)

	#Compute Solution
	u = Function(V)


	solve(a==L,u,bc)

	#Plot Solution
	if plotme:
		plot(mesh,interactive=True)
		plot(u,interactive=True)

	#Compute Error
	print("L2 Error: {:.2E}".format(errornorm(u_e,u,'L2')))
	print("H1 Error: {:.2E}".format(errornorm(u_e,u,'H1')))


if __name__ == "__main__":
	mu = 0.01
	my_solver(8,mu,'CG',1)
	my_solver(16,mu,'CG',1)
	my_solver(32,mu,'CG',1)
	my_solver(64,mu,'CG',1)
	my_solver(128,mu,'CG',1)
	my_solver(256,mu,'CG',1)



