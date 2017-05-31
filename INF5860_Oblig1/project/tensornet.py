
### IMPORT DATA ###

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)	#one-hot encoding: Each class is encoded as some collection of bits, where one of the bits are 1.

### START TENSORFLOW SESSION ###

import tensorflow as tf 

import numpy as np 

sess = tf.InteractiveSession()

### BUILD COMPUTATIONAL GRAPH FOR SOFTMAX REGRESSION MODEL ###

## INPUT/OUTPUT NODES

x = tf.placeholder( tf.float32, shape = [ None, 784] )  		#2d-tenor: single, flattened 28 x 28 image, with batch size = None

y_ = tf.placeholder( tf.float32, shape = [ None, 10] )			#2d-tensor: one-hot 10 dimensional vector indicating which digit class 0-9 image belongs to

## VARIABLES

W = tf.Variable( tf.zeros( [ 784, 10 ] ) )						#Weights, a variable is a value that lives in Tensorflow's computation graph, can be used and modified by computations.

b = tf.Variable( tf.zeros( [ 10 ] ) )							#Bias, note that we have 784 weights, 1 for each image pixel, and 1 bias for each class, with 10 classes.

sess.run( tf.global_variables_initializer() )					#initialize variables in current session so they can be used, assigns the initial values to the variabels.

## IMPLEMENT REGRESSION MODEL

y = tf.matmul( x, W ) + b 										#sets y to be the matrix product W.T*x + b  ==> linear regression

cross_entropy = tf.reduce_mean(									#define loss function to be the stable formulation of the cross_entropy loss for softmax		
	tf.nn.softmax_cross_entropy_with_logits( 					#tf.nn.softmax_... applies softmax on unormalized prediction and sums accros all classes
		labels = y_, logits = y ) )								#reduce mean takes the average over these sums


## DEFINE HOW TO TRAIN THE MODEL

train_step = tf.train.GradientDescentOptimizer( 0.5 ).minimize( cross_entropy ) #Uses gradient-descent to minimize loss, with steps of 0.5

### TRAIN THE MODEL ###

for _ in range(1000):

	batch = mnist.train.next_batch( 100 )							#load 100 training examples at a time

	train_step.run( feed_dict = { x : batch[0], y_ : batch[1] } ) 	#apply gradient_descent, feed_dict replaces the placeholder tensors with the training_examples

### EVALUATE MODEL ###

correct_prediction = ( tf.equal( tf.argmax( y, 1 ), tf.argmax(y_,1) ) ) 	#tf_argmax( y , 1) is the label our model thinks is most likely, tf_argmax(y_,1) is the true label,
																			# we obtain a list of booleans that is True where the classes are the same

accuracy = tf.reduce_mean( tf.cast( correct_prediction, tf.float32 ) )		#casts the boolean values to floats and takes the mean.

print( accuracy.eval( feed_dict = {x : mnist.test.images, y_ : mnist.test.labels} ) )  #prints the accuracy on test set, feed_dict replaces x and y by test images and test labels