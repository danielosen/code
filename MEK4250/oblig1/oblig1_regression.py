import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#step-size
h = -np.log([8,16,32,64,128,256])

#using error obtained from exercise 1 -first order
l2_error_k_1 = np.log([3.28E-02,8.46E-03,2.13E-03,5.34E-04,1.34E-04,3.34E-05])
h1_error_k_1 = np.log([4.36E-01,2.18E-01,1.09E-01,5.45E-02,2.73E-02,1.36E-02])

l2_error_k_10 = np.log([6.77E-01,3.63E-01,1.78E-01,5.49E-02,1.45E-02,3.67E-03])
h1_error_k_10 = np.log([2.55E+01,1.72E+01,1.05E+01,5.43E+00,2.72E+00,1.36E+00])

print('first order exerc1:')
slope,intercept,_,_,_ = stats.linregress(h,l2_error_k_1)
print("l2k1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_k_1)
print("h1k1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_k_10)
print("l2k10, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_k_10)
print("h1k10, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))

#using error obtained from exercise 1 -second order
l2_error_k_1 = np.log([5.69E-04,6.93E-05,8.61E-06,1.08E-06,1.34E-07,1.68E-08])
h1_error_k_1 = np.log([3.31E-02,8.39E-03,2.11E-03,5.27E-04,1.32E-04,3.30E-05])

l2_error_k_10 = np.log([4.24E-01,8.86E-02,1.02E-02,1.14E-03,1.37E-04,1.69E-05])
h1_error_k_10 = np.log([1.77E+01,6.72E+00,1.96E+00,5.17E-01,1.31E-01,3.29E-02])

print('second order exerc1:')
slope,intercept,_,_,_ = stats.linregress(h,l2_error_k_1)
print("l2k1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_k_1)
print("h1k1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_k_10)
print("l2k10, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_k_10)
print("h1k10, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))


#using error obtained from exercise 2 -first order
l2_error_mu1 = np.log([1.40E-03,3.51E-04,8.77E-05,2.19E-05,5.48E-06,1.37E-06])
l2_error_mu01 = np.log([2.37E-02,6.18E-03,1.56E-03,3.91E-04,9.79E-05,2.45E-05])
l2_error_mu001 = np.log([2.39E-01,1.04E-01,3.81E-02,1.13E-02,2.96E-03,7.51E-04])

h1_error_mu1 = np.log([3.75E-02,1.88E-02,9.38E-03,4.69E-03,2.35E-03,1.17E-03])
h1_error_mu01 = np.log([7.69E-01,3.98E-01,2.01E-01,1.01E-01,5.04E-02,2.52E-02])
h1_error_mu001 = np.log([7.80E+00,7.01E+00,5.09E+00,2.98E+00,1.57E+00,7.94E-01])

print('first order exerc2:')
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu1)
print("l2mu1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu1)
print("h1mu1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu01)
print("l2mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu01)
print("h1mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu001)
print("l2mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu001)
print("h1mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))

#using error obtained from exercise 2 -second order
l2_error_mu1 = np.log([1.15E-05,1.45E-06,1.82E-07,2.28E-08,2.85E-09,3.56E-10])
l2_error_mu01 = np.log([2.25E-03,3.04E-04,3.88E-05,4.89E-06,6.12E-07,7.66E-08])
l2_error_mu001 = np.log([8.67E-02,3.08E-02,7.65E-03,1.33E-03,1.86E-04,2.40E-05])

h1_error_mu1 = np.log([5.97E-04,1.50E-04,3.77E-05,9.45E-06,2.36E-06,5.91E-07])
h1_error_mu01 = np.log([1.19E-01,3.17E-02,8.07E-03,2.03E-03,5.08E-04,1.27E-04])
h1_error_mu001 = np.log([5.63E+00,3.80E+00,1.74E+00,5.69E-01,1.56E-01,3.99E-02])

print('second order exerc2')
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu1)
print("l2mu1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu1)
print("h1mu1, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu01)
print("l2mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu01)
print("h1mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu001)
print("l2mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu001)
print("h1mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))

#using SUPG  -first order
h = -np.log([8,16,32,64,128])

l2_error_mu01 = np.log([1.17E-1,6.33E-2,3.31E-2,1.70E-2,8.60E-03])
l2_error_mu001 = np.log([2.01E-1,1.32E-1,7.99E-2,4.55E-2,2.47E-02])

h1_error_mu01 = np.log([1.01,6.22E-1,3.52E-1,1.88E-1,9.75E-2])
h1_error_mu001 = np.log([5.43,5.79,4.97,3.63,2.32])

print('first-order order supg')
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu01)
print("l2mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu01)
print("h1mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu001)
print("l2mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu001)
print("h1mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))

#using SUPG -second order
l2_error_mu01 = np.log([ 5.94E-2,2.06E-2,6.22E-3,1.73E-3,4.56E-4])
l2_error_mu001 = np.log([ 1.77E-1,1.09E-1,5.79E-2,2.56E-2,9.35E-3])

h1_error_mu01 = np.log([ 5.63E-1,2.23E-1,7.09E-2,2.00E-2,5.30E-3])
h1_error_mu001 = np.log([ 5.43,5.16,3.87,2.25,9.75E-1])

print('second-order order supg')
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu01)
print("l2mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu01)
print("h1mu01, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,l2_error_mu001)
print("l2mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
slope,intercept,_,_,_ = stats.linregress(h,h1_error_mu001)
print("h1mu001, slope= {:.2E}, exp-intercept = {:.2E}").format(slope,np.exp(intercept))
