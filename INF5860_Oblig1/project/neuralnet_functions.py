import numpy as np 

def sigmoid(data):
	return 1.0/(1.0+np.exp(-data))

def sigmoid_gradient(y):
	y = sigmoid(y)
	return y*(1-y)

def relu(x):
	return np.max(np.zeros(shape=x.shape),x)

def relu_gradient(x):
	y = np.ones(shape=x.shape)
	return y[x>=0]

def identity(x):
	return x 

def identity_gradient(y):
	return np.ones(shape=y.shape)

def l2_loss(y_exact,y_pred):
	return 0.5*np.dot((y_exact-y_pred).T ,y_exact-y_pred)

def l2_loss_gradient(y_exact,y_pred):
	return y_pred-y_exact

def l1_loss(y_exact,y_pred):
	return np.sum(np.abs(y-exact-y_pred))

def l1_loss_gradient(y_exact,y_pred):
	pass

def softmax_loss(y_exact,y_pred):
	pass