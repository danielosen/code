"""
Used to test addition_testing.py
"""
import sys
import collections
import os
sys.path.insert(0, os.path.join(os.environ['HOME'], 'python', 'lib'))

class UnitTest(object):
    def __init__(self, func, args, kwargs, res):    # make test
        try:
        	self.answer = func(args[0],args[1],kwargs["num_rechecks"])
        	self.realanswer = res    
       	except:
       		self.answer = False
       		self.realanswer = True

    def __call__(self):                             # run test
        return self.answer == self.realanswer