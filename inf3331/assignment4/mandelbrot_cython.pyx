#Mandelbrot_cython.pyx
#Same method as with SWIG, except now we can make the image without numpy
#complex and pow are among the built-in constants and functions, so they are not needed to declare
#as evidenced by: http://cython.readthedocs.io/en/latest/src/userguide/language_basics.html
#The complex unit can be called with complex(0,1.0), 1j,
#We can numpy arrays, but apparently, this is slower than so-called memoryviews
#however, it is much, much easier to just work with numpy arrays, though overhead may be larger
#sizeof double complex and complex are the same... So a complex is always a double complex importnumpy
#we assume the image is empty when we get it, and set the zeros ourselves
#install: python3 setup_cython.py build_ext --inplace

#cimport numpy
from libc.stdlib cimport malloc, free
from libc.math cimport pow

cdef extern from "complex.h":
	double creal(complex arg)
	double cimag(complex arg)


cpdef generate_Image(unsigned short[:,:] image, double a, double b, double c, double d, int Ny, int Nx, int escapetime):
	cdef complex complex_I = complex(0,1) #numpy's complex64_t is sizeof 8, while c complex is sizeof 16, use 128_t
	#allocate new matrix
	cdef complex **z = <complex **>malloc(Ny * sizeof(complex*))
	cdef int i,j,t
	cdef double radius = 4.0
	cdef int lenx = Nx-1
	cdef int leny = Ny-1
	#SANITY CHECKING
	if lenx <= 0:
		lenx = 1
		b = a
	if leny <= 0:
		leny = 1
		d = c
	#Iterate from a,d in the top left to b,c in the bottom right
	for i in range(0,Ny):
		z[i] = <complex *>malloc(Nx * sizeof(complex))
		for j in range(0,Nx):
			z[i][j] = a+(b-a)/(lenx)*j + (d+(c-d)/(leny)*i)*complex_I
			image[i,j] = 0
	for i in range(0,Ny):
		for j in range(0,Nx):
			for t in range(0,escapetime):
				if pow(creal(z[i][j]),2) + pow(cimag(z[i][j]),2) <= radius:
					z[i][j] = z[i][j]*z[i][j]+ a+(b-a)/(lenx)*j + (d+(c-d)/(leny)*i)*complex_I
					image[i,j] += 1
				else:
					break

	for i in range(0,Ny):
		free(z[i])
	free(z)
	pass


