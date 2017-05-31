%module mandelbrot_generator

%{
	#define SWIG_FILE_WITH_INIT
	#include "mandelbrot_generator.h"
%}


%include "numpy.i"
%init %{
	import_array();
%}

%apply (unsigned short int* INPLACE_ARRAY2, int DIM1, int DIM2) {(unsigned short int* image, int dimy, int dimx)}
%include "mandelbrot_generator.h"

/* usage: swig -c++ -python mandelbrot_generator.i */
/* usage: python setup.py build_ext --inplace*/ 
/* OR IF PYTHON3: python3 setup.py build_ext --inpalace*/
/* import mandelbrot_generator */