import numpy as np
import matplotlib.pyplot as plt 


def linregpredict(X,thetavec):
	# Use the training data X and the current estimate thetavec to predict yhat
	# Hint: use .dot to predict all samples in one operation
	return X*thetavec

def linregloss(y,yhat):
	#returns mean square error
	return (yhat-y).T*(yhat-y)/(2*y.size)

def gradientdescent(yhat,y,epsilon,thetavec,X):
	#update thetavec using gradient descent
	return thetavec - epsilon*X.T*(yhat-y)

def normalequations(X,y):
	thetavec = np.linalg.inv(X.T*X)*X.T*y
	return thetavec

# Define training set
X = np.matrix([[1,1],[1,2],[1,3]])
y = np.matrix([[1],[2],[2.5]])

# best solution
thetavec_true = normalequations(X,y)


# Init method
epsilon = 0.01
thetavec = np.matrix([[0],[1]])

# Loop to find optimal weights
error = [linregpredict(X,thetavec).item(0)]
error_theta = [linregloss(thetavec_true,thetavec).item(0)]
n = [0]
for i in range(1,1000):
	yhat = linregpredict(X,thetavec)
	thetavec = gradientdescent(yhat,y,epsilon,thetavec,X)
	error.append(linregloss(y,yhat).item(0))
	n.append(i)
	error_theta.append(linregloss(thetavec_true,thetavec).item(0))

plt.plot(n,error)
plt.show()
plt.plot(n,error_theta)
plt.show()


