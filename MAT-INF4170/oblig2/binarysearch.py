import numpy as np 


def find_mu(x,t):
	'''not used'''
	for mu in range(len(t)-2,0,-1):
		if (t[mu] <= x):
			return mu
	else:
		return 0

def binary_find_mu(x,t):
	'''returns -1 for the endpoint case'''
	first = 0
	last = t.size-2
	found = False
	mu = -1

	while first <= last and not found:
		midpoint = (first+last)//2
		if t[midpoint] <= x and x < t[midpoint+1]:
			mu = midpoint
			found = True
		else:
			if x < t[midpoint]:
				last = midpoint - 1 
			else:
				first = midpoint + 1 
	return mu

def find_mu_v2(x,t):
	'''note used'''
	return t[t <= x].argmax()


if __name__ == '__main__':
	'''run this script to compare near-worst case runtime'''
	import time
	t = np.linspace(0.0,1.0,num=10000)
	x = t[1]+10**(-15)

	start = time.time()
	print find_mu(x,t)
	end = time.time()
	print end-start

	start = time.time()
	print binary_find_mu(x,t)
	end = time.time()
	print end-start

	start = time.time()
	print find_mu_v2(x,t)
	end = time.time()
	print end-start

