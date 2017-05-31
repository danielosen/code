from __future__ import division

from sklearn import datasets


import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline
sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import math


data = datasets.load_iris()
X = data.data[:100, :2]
y = data.target[:100]
X_full = data.data[:100, :]


#setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
#versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
#plt.xlabel("Sepal Length")
#plt.ylabel("Sepal Width")
#plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
#plt.show()

nofsamp,nfeat = X.shape
Xappend = np.ones((nofsamp,nfeat+1))
Xappend[:,1:] = X

#print Xappend[1:10,:]

#def logistic_func(theta, x):
#    return float(1) / (1 + math.e**(-x.dot(theta)))

def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc

def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)


def grad_desc(theta_values, X, y, lr=.001, converge_change=.001):
    #normalize
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
    return theta_values, np.array(cost_iter)
def pred_values(theta, X):
    #normalize
   # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)

    return pred_value

shape = Xappend.shape[1]


#Predict sigmoid(0)
def test_logistic_func():
    from logreg_noregularization import logistic_func
    x0vec = np.zeros((5,nfeat))
    X0append = np.ones((5,nfeat+1))
    X0append[:,1:]=x0vec

    theta0 = np.zeros((nfeat+1))
    hzero = logistic_func(theta0, X0append)
    hmean = np.mean(hzero)

    correctval = 0.5;
    assert np.abs(hmean-correctval)<1e-2



def test_cost_func():
    from logreg_noregularization import cost_func
    Xtest = Xappend[:10,:]
    #x0vec = np.zeros((5, nfeat))
    #X0append = np.ones((5, nfeat + 1))
    #X0append[:, 1:] = x0vec
    ytest = y[:10]

    thetatest = np.zeros(nfeat+1)
    thetatest[0]=0.0
    thetatest[1] = 1.0
    thetatest[2]= 1.0
    testcost = cost_func(thetatest, Xtest, ytest)
    correctcost = 8.17
    assert np.abs(testcost-correctcost)<0.5

def test_log_gradient():
    from logreg_noregularization import log_gradient
    Xtest = Xappend[:10, :]

    ytest = y[:10]
    thetatest = np.zeros(nfeat + 1)
    thetatest[0] = 0.0
    thetatest[1] = 0.0
    thetatest[2] = 0.0
    testthetagrad = log_gradient(thetatest, Xtest, ytest)
    correcttheta = np.zeros((nfeat+1))
    #print testthetagrad
    correcttheta[0] = 5.0
    correcttheta[1]=24.3
    correcttheta[2]=16.55
    diff = np.abs(testthetagrad-correcttheta)
    assert np.mean(diff)<0.1


#y_flip = np.logical_not(y) #flip Setosa to be 1 and Versicolor to zero to be consistent
#betas = np.zeros(shape)
#fitted_values, cost_iter = grad_desc(betas, Xappend, y_flip)
#print(fitted_values)






#predicted_y = pred_values(fitted_values, Xappend)
#predicted_y

#correct = np.sum(y_flip == predicted_y)
#print('correc', correct)

#plt.plot(cost_iter[:,0], cost_iter[:,1])
#plt.ylabel("Cost")
#plt.xlabel("Iteration")
#plt.show()