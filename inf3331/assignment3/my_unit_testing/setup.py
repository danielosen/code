#python 2.7
from distutils.core import setup
name='my_unit_testing'

setup(name=name,
      version='0.1',
      py_modules=[name],       # modules to be installed
      scripts=[name + '.py'],  # programs to be installed
      )