#!/usr/bin/env python

#python3 setup.py build_ext --inplace

#Run this to finish installing the mandelbrot generator
#once the c++ files have been wrapped with SWIG
#see mandelbrot_generator.i
#You WILL need numpy.i
#If you somehow didn't get that with this folder,
#you can find it at numpy's github
#in folders .../tools/swig/numpy.i

"""
setup.py file for SWIG mandelbrot_generator

"""
import numpy as np
from distutils.core import setup, Extension


mandelbrot_generator = Extension('_mandelbrot_generator',
                           sources=['mandelbrot_generator.cpp','mandelbrot_generator.i'],
                           swig_opts=['-c++','-py3'],
                           include_dirs=[np.get_include()],
                           )

setup (name = 'Mandelbrot Generator',
       version = '0.1',
       author      = "Daniel Osen",
       description = """Mandelbrot Generator SWIG""",
       ext_modules = [mandelbrot_generator],
       py_modules = ["mandelbrot_generator","test_mandelbrot"],
       )