import numpy as np 
import matplotlib.pyplot as plt 

import neuralnet,neuralnet_functions

def test_neuralnet():
	'''tests neuron module for a simple function f(x,y,z) = 1+x+y+z

	Each data point is assumed to be a column-vector in the data matrix.
	Hence data = [ x_1, x_2, x_3 ; y_1, y_2, y_3,; z_1, z_2, z_3],
	each point being the column-vector [x_i;y_i;z_i].
	
	We also assume weights are stored as a column-vector.

	Hence the linear function is evaluated weights.T*data

	i.e. (3 x 1 ).T * (3 x 3) = 1 x 3 (output is a row vector)

	for each column (3 x 1).T * (3 x 1) = 1 x 1

	This means we have to do for loops column wise not row-wise in data.

	Unfortunately, when you get a row or column vector out of a matrix in numpy
	the dimension information is lost and the resulting vector is always a row vector...

	'''


	#store exact_answer
	x = np.random.randn(100)
	y = np.random.randn(100)
	z = np.random.randn(100)
	bias = 0
	data = np.matrix([x,y,z])
	weights = np.matrix([[1],[2],[3]])
	exact_f = weights.T*data+bias


	### BUILD NEURAL NET

	# define neuralnet with L2 loss and give it the training output to compare with
	my_neuralnet = neuralnet.neuralNet(neuralnet_functions.l2_loss, neuralnet_functions.l2_loss_gradient)

	# define neuron with identity activation function (pure linear regression)
	my_neuron0 = neuralnet.linearNeuron(3, neuralnet_functions.identity, neuralnet_functions.identity_gradient) 
	my_neuron1 = neuralnet.linearNeuron(3, neuralnet_functions.identity, neuralnet_functions.identity_gradient) 
	my_neuron2 = neuralnet.linearNeuron(2, neuralnet_functions.identity, neuralnet_functions.identity_gradient)
	
	#insert neuron at first layer
	my_neuralnet.insert_neuron(my_neuron0,0)
	my_neuralnet.insert_neuron(my_neuron1,0)
	my_neuralnet.insert_neuron(my_neuron2,1)

	### TRAIN NEURAL NET
	learning_rate = 0.01
	epoch_vec,loss_vec = my_neuralnet.train(data,exact_f.T,learning_rate,100)
	plt.plot(epoch_vec,loss_vec)
	print(loss_vec[-1])
	plt.show()




test_neuralnet()

#y =np.array([1,2,3])
#x = np.array([3,4,5])
#print( (y-x).T*(y-x) )
#print np.dot((y-x).T,(y-x))
#print np.dot(y.T,y)
