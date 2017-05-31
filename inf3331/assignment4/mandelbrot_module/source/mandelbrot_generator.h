// mandelbrot_generator header file 
#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <complex.h>
#include <stdbool.h>

//import_array();

void generate_Image(unsigned short int* image,int dimy, int dimx, double a, double b, double c, double d,int escapetime);



//bool is_feasible(double a, double b, double c, double d);