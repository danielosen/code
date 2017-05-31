import numpy as np

class computationalGraph(object):
	pass

class lossGate(object):
	pass


class neuralNet(object):
	'''contains fully-connected (acylic) neurons in layers'''
	def __init__(self,loss_function,loss_gradient):
		self.loss_function = loss_function
		self.loss_gradient = loss_gradient
		self.neurons = []

	def insert_neuron(self,neuron,layer):
		if 0<= layer and layer < len(self.neurons):
			self.neurons[layer].append(neuron)
		else:
			if layer==0:
				self.neurons.append([])
			else:
				for i in range(layer):
					self.neurons.append([])
			self.neurons[layer].append(neuron)

	def forward(self,data):
		'''Assumes data matrix has column-vector features for each point.
		Numpy has a very bad feature: if you get a column or row from matrix,
		the result is always a row vector... So we have to transpose it.'''

		#iterate over columns in data
		for column in range(0,data.shape[1]):
			
			column_data = data[:,column]

			#Iterate over layers
			for neurons in self.neurons:
				
				#Each layer has an output, that must be passed to all the neurons in the next layer,
				#we store this temporarily. The output size of a layer will be a column-vector of size num_neurons,
				#since each neuron outputs a 1x1 number

				num_neurons = len(neurons)

				layer_output = np.matrix(np.zeros(shape=(num_neurons,1)))

				#iterate over neurons in layers to forward data
				for i in range(num_neurons):
					
					layer_output[i] = neurons[i].forward(column_data)

				#update the data for the next layer
				column_data = layer_output

			#when we have finished forwarding the data through all the layers, yield the final output to be computed as loss
			yield column_data

				


	def backward(self,gradient,learning_rate):
		#gradient input is gradient w.r.t to loss

		#iterate over layers in reverse order
		for neurons in reversed(self.neurons):

			num_neurons = len(neurons)

			#Each layer may have graadients of different sizes. For instance,
			#coupling two neurons output to one neuron, then this neuron has a gradient of size 2,
			#since it has two inputs. The previous neurons may take in data of size n.
			#Because of this, either the input layer must be specified as n neurons doing one multiplication each
			#or we need to keep track of what neurons take in input of other neurons and the data size.
			#being lazy, just make a dynamic list.

			layer_gradient = []
			for i in range(num_neurons):
				#iterate over neurons in layer, obtain row vector gradient
				for column_grad in gradient.T:
					column_grad = column_grad.T 
					layer_gradient.append(neurons[i].backward(column_grad,learning_rate))
			
			#Each row-vector in the layer-gradient is a neurons gradient w.r.t to all of the input it received.
			#in particular, if its input was x,y,z, that means there were three neurons, each computing either x or y and z
			#in the previous layer. Since we made into a row-vector, then since we have a fully-connected net
			#all of the x,y,z are going to all of the neurons at the layer we are computing the gradient. So if the current
			#layer had output u,v,w, then 

			#x,y,z -> u, v , w
			'''layer_gradient:		[du/dx, du/dy, du/dz]
									 dv/dx, dv/dy  dv/dz
									 dw/dx  dw/dy  dw/dz
			Hence each column of this is the gradient we want for each of the neurons in the previous layer.
			So the neuron responsible for calculating x, receives gradient [du/dx,dv/dx,dw/dx]
			'''
			gradient = np.matrix(np.zeros(shape=(num_neurons,layer_gradient[0].shape[1])))
			for i in range(num_neurons):
				gradient[i][:] = layer_gradient[i]


	def train(self,data,training_output,learning_rate,max_epochs):

		'''We train by epochs,each epoch is forwarding the data,
		then backpropagating the gradient and updating weights
		For purposes of training, we iterate through all of the data once,
		update for each feature column vector, then finish the first epoch,
		and repeat the process on the same data.'''

		epoch = 0
		loss_vec = np.zeros(max_epochs)
		epoch_vec = np.zeros(max_epochs)
		while (epoch < max_epochs):
			i = 0
			for output in self.forward(data):

				'''compute the loss after forwarding data to final output,
				comparing with given training_output'''

				loss = self.loss_function(training_output[i],output)

				

				'''compute the gradient w.r.t to the output, NOT training output! since we cannot change this.
				'''

				gradient = self.loss_gradient(training_output[i],output)


				self.backward(gradient,learning_rate)
				i+= 1
			loss_vec[epoch]= loss.item(0,0)
			epoch_vec[epoch] = epoch
			epoch += 1

			#check if loss not improving -> try to reduce learning_rate
			if epoch >= 2:
				if (loss_vec[-1]-loss_vec[-3]) < 2*learning_rate:
					learning_rate *= 0.99

		print "weights = {}".format((self.neurons[0][0]).weights)
		print "weights = {}".format((self.neurons[1][0]).weights)
		return epoch_vec[0:epoch],loss_vec[0:epoch]

class lossNode(object):
	pass

class inputNode(object):
	pass


class linearNeuron(object):
	'''During forward propagation, takes some input, multiplies it with weights, and returns the activation value.
	During backward propagation, takes some gradient at an epoch, multiplies it with its own gradient at the epoch,
	and returns the result. The neuron computes a linear combination of the input data and weights.
	Assumes 1D data of matrix type with column-vector features! Weights are stored as columns.'''

	def __init__(self,n,activation_function,activation_gradient,halved_activation = False):
		'''Xavier Initialization, modified to account for reLU (halved activation)'''
		self.weights = np.matrix(np.random.randn(n,1)) #draws from the normal distribution
		if halved_activation:
			self.weights /= np.sqrt(n/2)
		else:
			self.weights /= np.sqrt(n)
		self.activation_function = activation_function
		self.activation_gradient = activation_gradient

	def update(self,learning_rate,weights_delta):
		self.weights += -learning_rate*weights_delta

	def forward(self,data):
		'''Given some 1D data, outputs activation value using linear combination of data and weights, and an activation function.'''
		self.data = data
		return self.activation_function(self.weights.T*self.data)

	def backward(self,gradient,learning_rate):
		'''computes gradient at current epoch, requires at least one forward propagation to make sense.

		The gradient is the the loss gradient at the previous layer, i.e. the derivative of the loss w.r.t.
		to the neuron parameters/functions in the  previous layer. Since a linearNeuron computes:
		z = w.T*x, and  f(z), it's derivative w.r.t to weights w is always df/dz*dz/dw = df/dz*x.
		In this case, we can choose to obtain row gradients, given a column gradient.

		Since the neuron now knows what effect it has on the the loss, it can update its weights
		immediately using df/dz*x It can also backpropagate its change on the loss w.r.t to the input it receives,
		i.e. the change df/dz*w.T. To be sure, this backpropagation should be done with the old weights
		'''
		weights_delta = (gradient*self.activation_gradient(self.weights.T*self.data)*self.data.T).T

		old_weights = self.weights

		self.update(learning_rate,weights_delta)

		return gradient*self.activation_gradient(self.weights.T*self.data)*old_weights.T


