#MEK4250 Oblig 1

#IMPORTS
from dolfin import*
from numpy import pi


#DEFS 
def my_solver(n,k,element_family,degree,plotme=False):

	def boundary(x,on_boundary):
		if on_boundary:
			if near(x[0],0.0) or near(x[0],1.0):
				return True
		return False

	#Generate Mesh (0,1)^2
	mesh = UnitSquareMesh(n,n)

	#Define Variational Problem
	V = FunctionSpace(mesh,element_family,degree)
	u = TrialFunction(V)
	v = TestFunction(V)
	f = Expression('2*pow(pi,2)*pow(k,2)*sin(k*pi*x[0])*cos(k*pi*x[1])',k=k,pi=pi)
	u_e = Expression('sin(k*pi*x[0])*cos(k*pi*x[1])',k=k,pi=pi)
	a = dot(grad(u),grad(v))*dx
	L = f*v*dx

	#Define dirichlet boundary condition
	bc = DirichletBC(V,Constant(0.0),boundary)

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
	my_solver(8,10,'CG',2)
	my_solver(16,10,'CG',2)
	my_solver(32,10,'CG',2)
	my_solver(64,10,'CG',2)
	my_solver(128,10,'CG',2)
	my_solver(256,10,'CG',2)



