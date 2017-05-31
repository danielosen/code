from __future__ import division

from sklearn import datasets


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import math


data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]

nofsamp,nfeat = X.shape
Xappend = np.ones((nofsamp,nfeat+1))
Xappend[:,1:] = X



# Remember to add a column of ones to X, as the first column
'''

setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
plt.show()
'''



def logistic_func(theta, x):
    #Fill in the correct value for the sigmoid function, the logistic function, not logarithm
    # If the input x is a vector, the output should be a vector

    sigmoidvalue = 1/(1+np.exp(-np.dot(x,theta)))
    print(x)
    print(theta)

    return sigmoidvalue

def test_logistic_func():
    x0vec = np.zeros((5,nfeat))
    X0append = np.ones((5,nfeat+1))
    X0append[:,1:]=x0vec

    theta0 = np.zeros((nfeat+1))
    hzero = logistic_func(theta0, X0append)
    hmean = np.mean(hzero)

    correctval = 0.5
    assert np.abs(hmean-correctval)<1e-2



def log_gradient(theta, x, y):
     #Compute the gradient of theta to use in gradient descent updates, without learning rate
     #All nfeat elements in theta should be

    return theta_gradient


def cost_func(theta, x, y):
    # Compute the cost function for logistic regression

    return costval


def grad_desc(theta_values, X, y, lr=.01, converge_change=.001):
    #Do gradient descent with learning rate lr and stop of the nof. changes is below limit
    #Return the resulting theta values, and an array with the cost values for each iteration
    # Stop if the abs(cost(it)-cost(it+1))<convergence_change

    return theta_values, cost_function_array


def pred_values(theta, X):
    '''Predict the class labels'''


    return pred_value


test_logistic_func()
'''
#X should be with an extra column added
shape = X.shape[1]
y_flip = np.logical_not(y) #flip Setosa to be 1 and Versicolor to zero to be consistent
betas = np.zeros(shape)
fitted_values, cost_iter = grad_desc(betas, X, y_flip)
print(fitted_values)
# Your theta-vector should be about 0.20 -1.22 2.07

predicted_y = pred_values(fitted_values, X)
predicted_y

correct = np.sum(y_flip == predicted_y)
print('correc', correct)

plt.plot(cost_iter[:,0], cost_iter[:,1])
plt.ylabel("Cost")
plt.xlabel("Iteration")
plt.show()
'''