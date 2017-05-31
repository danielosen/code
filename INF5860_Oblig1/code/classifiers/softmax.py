import numpy as np
from random import shuffle

### Some comments about this task. There are multiple definitions of the softmax loss function,
### and they usually all differ in what constants you multiply them with...
### I have chosen, and this is from discussion in lectures...,
### the exact formulas available from: http://cs231n.github.io/neural-networks-case-study/
### 1/M * softmax_loss, where M = amount of data, and penalty = reg constant * sum of squared weights
### i do not penalize bias weights, and the definition from that page doesnt divide penalty by number of weights.
### however, although it might introduce further instabilities from division, it actually helps us keep
### the loss function and penalty reasonably close in magnitude, which helps with vizualisation.

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """


  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)


  ############################################################################
  #Softmax Regression                                                        #
  #Source: http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression       #
  #        http://cs231n.github.io/linear-classify/#softmax                  #
  ############################################################################

  ### COMPUTE SOFTMAX LOSS AND GRADIENT (NO REGULARIZATION) ###
  #NOTE: For the purposes of using explicit loops, i only consider the outer sums.
  
  #get outer sum indices
  m = X.shape[0]
  k = W.shape[1]
  #loop over outer sum
  for i in range(m):
    for j in range(k):
      #compute loss and gradient for softmax loss
      #dividing large numbers can be numerically unstable, add scaling factor
      logC = -np.max(np.dot(X[i,:],W[:,:]))
      numerator = np.exp(np.dot(X[i,:],W[:,j])+logC)
      denominator = np.sum(np.exp(np.dot(X[i,:],W[:,:])+logC))
      loss += float(y[i]==j)*np.log(numerator/denominator)
      dW[:,j] += X[i,:]*(float(y[i]==j)-numerator/denominator)
  #scale loss and gradient
  loss *= -1.0/float(m)
  dW *= -1.0/float(m)
  ### COMPUTE SOFTMAX LOSS AND GRADIENT (REGULARIZATION ONLY) ###
  #NOTE: I am using a scaled frobenius norm (raised to power 2) regularization of params, does not regularize bias params.
  #initialize regularization penalty and gradient for penalty
  penalty = 0.0
  penalty_grad = np.zeros_like(W)
  #loop over elements in W which are not bias weights;
  #since the linear combination is np.dot(X,W), and the ones are at X[:,-1],
  #these weights are W[-1,:], so we skip the last row
  p = W.shape[0]
  for i in range(p-1):
    for j in range(k):
      penalty_grad[i,j] += W[i,j]
      penalty += W[i,j]**2
  #scale penalty by number of weights penalized, take the square root since it is a norm
  penalty = reg/(2.0*(float(W.shape[0]-1)*float(W.shape[1])))*penalty 
  penalty_grad *= reg/(1.0*float(W.shape[0]-1)*float(W.shape[1])) #the 1/2 disappears when taking the derivative
  #add penalty,penalty_grad to loss and grad
  loss += penalty
  dW += penalty_grad

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """

  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  ############################################################################
  #Softmax Regression                                                        #
  #Source: http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression       #
  #        http://cs231n.github.io/linear-classify/#softmax                  #
  ############################################################################

  ### COMPUTE VECTORIZED SOFTMAX LOSS AND GRADIENT ###
  #get outer summation index
  m = X.shape[0]
  k = W.shape[1]
  #get linear combination
  yhat = np.dot(X,W)
  #Make a NxC matrix, where for each data point i (row) the correct class y[i]==j (column) is 1, and 0 elsewhere
  #this is equivalent to a one-hot encoding of y
  yhot = np.zeros_like(yhat)
  y_i = np.arange(m)
  yhot[y_i,y] = 1
  #ordinarly we would compute:
  #numerator = np.exp(yhat[y_i,y]+logC)
  #denominator = np.sum(np.exp(yhat+logC),axis=1)
  #to obtain loss and gradient as before
  #instead broadcast row-wise copy of denominators into W.shape[1] columns and divide elementwise in yhat to obtain a matrix with num/denom for each i,j
  #this guy works ok! error is to machine precision compared to for-loops, its not neccessary to do this for loss,
  #but i couldnt figure out a better way for the gradient..
  logC = -np.max(np.max(yhat))
  num_denum = np.divide(np.exp(yhat+logC),np.dot(np.sum(np.exp(yhat+logC),axis=1).reshape(m,1),np.ones(shape=(1,k))))
  #compute the loss with regularization
  #only grab where y[i] == j from num/denum
  loss += -(1.0/float(m))*np.sum(np.sum(np.log(num_denum[y_i,y])))
  #skip the bias params when computing loss
  loss += (reg/(2.0*(float(W.shape[0]-1)*float(W.shape[1]))))*np.sum(np.sum(W[0:-1,:]*W[0:-1,:]))
  #After much headache, dot the X with the yhot, to only obtain the sums of X[i,:] where y[i]==j
  #and with the num/denum terms for other i,j
  dW += -np.dot(X.T,num_denum)
  dW += np.dot(X.T,yhot)
  dW *= -1.0/float(m)
  #finally add the vectorized gradient from the regularization, still not regularizing bias params (as before the 1/2 disappears on derivation)
  dW[0:-1,:] += reg/( 1.0*float(W.shape[0]-1)*float(W.shape[1] ))*W[0:-1,:]

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def simple_test():
  #tests that vectorization and naive agree (they do)
  D = 3
  N = 10
  C = 5
  W = np.random.randn(D,C)
  X = np.random.randn(N,D)
  y = np.random.randint(0,C,size=N)
  reg = 1.50
  loss1,dW1 = softmax_loss_naive(W,X,y,reg)
  loss2,dW2 = softmax_loss_vectorized(W,X,y,reg)
  assert ( np.allclose(dW1,dW2) )
  assert ( np.allclose(loss1,loss2) )

