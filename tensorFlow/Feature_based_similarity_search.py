"""
This file contained the feature-based similarity search method

Created by Zexi Chen(zchen22)
Date: Oct 2, 2016
"""

import numpy
import six.moves.cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

# read the time series data from file
def loadData(path):
    fileObject = open(path, 'r')
    dataset = pickle.load(fileObject)

    return dataset

# define dtw function
def dtw(list1, list2, window = 1):
    len1 = len(list1)
    len2 = len(list2)
    mat = [[float('inf') for x in range(len2 + 1)] for y in range(len1 + 1)]
    mat[0][0] = 0
    for i in range(1,len1 + 1):
        if i - window <= 1:
            start = 1
        else:
            start = i - window
        
        if i + window <= len2:
            end = i + window
        else:
            end = len2
        for j in range(start, end + 1):
            cost = abs(float(list1[i - 1] - list2[j - 1]))
            mat[i][j] = cost + min(mat[i-1][j], mat[i][j-1],mat[i-1][j-1])
        
    return mat[len1][len2]

# define euclidean distance function 
def euclideanDist(list1,list2):
    distance = 0
    for x in range(len(list1)):
        distance += pow((list1[x]-list2[x]),2)
    return math.sqrt(distance)

class NeuNet(object):
    """
    build a one layer CNN network
    """
    def __init__(
        self,
        w_shape,
        b_shape,
        input,
        activation = tf.nn.sigmoid,        
        n_visible = 46,
        n_hidden =  10
    ):

        # input dimension
        self.n_visible = n_visible
        # output dimension
        self.n_hidden = n_hidden

        # Weight matrix shape
        if w_shape is None:
            self.w_shape = [n_visible/2, 1, 1, n_hidden]
        else:
            self.w_shape = w_shape

        #bias shape
        if b_shape is None:
            self.b_shape = [n_hidden]
        else:
            self.b_shape = b_shape

        if input is None:
            self.input = tf.placeholder(tf.float32, shape=[None, n_visible])
        else:
            self.input = input

        # initialize Weight matrix and bias vector
        self.W = self.weight_variable(self.w_shape)
        self.b = self.bias_variable(self.b_shape)

        # nonlinear transformation function
        self.activation = activation
        self.output = self.construct_hidden()

    # define the weight matrix, initialize randomly 
    # truncated_normal: output random values from a truncated normal distribution with value out of 2 sd dropped 
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # define the bias
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # specify the model we use and set up the paratemers
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,self.n_visible/2,1,1], padding='SAME')

    # construct hidden features
    def construct_hidden(self):
        x_ts = tf.reshape(self.input, [-1, self.n_visible, 1, 1])
        h_conv = self.activation(self.conv2d(x_ts, self.W) + self.b)
        h_conv_reshape = tf.reshape(h_conv, [-1, 2 * self.n_hidden])

        return h_conv_reshape

    # define the cost function
    def cost_function(self, dtw_dists):
        h_conv_reshape = self.construct_hidden()
        hidden_feature_diff = tf.sub(h_conv_reshape[:,:self.n_hidden], h_conv_reshape[:,self.n_hidden:])

        square_euclidean = tf.reduce_sum(tf.square(hidden_feature_diff), 1, keep_dims=True)

        cost = tf.reduce_mean(tf.square(tf.sub(dtw_dists, square_euclidean)))

        return cost


def run_model(n_hiddens = [10], learning_rate = [1e-2], training_epochs = 10000):
    """
        n_hidd1: number of nerons in each hidden layers
    """
    train_set = loadData('../theano/data/samples1')
    valid_set = loadData('../theano/data/samples2')
    test_set = loadData('../theano/data/samples3')

    # reshape the array, concatenate two time series as one training instance
    train_set_reshape = numpy.reshape(train_set, (train_set.shape[0]/2, train_set.shape[1]*2))
    valid_set_reshape = numpy.reshape(valid_set, (valid_set.shape[0]/2, valid_set.shape[1]*2))
    test_set_reshape = numpy.reshape(test_set, (test_set.shape[0]/2, test_set.shape[1]*2))

    # re-scale input data
    train_set1 = train_set_reshape/255.0
    valid_set1 = valid_set_reshape/255.0
    test_set1 = test_set_reshape/255.0

    # calculate the squared dtw distance between the two time series in each row of the training data validation data and test data 
    # the dtw is used in the cost function as the target value to minimize.
    train_dtws = numpy.zeros((train_set1.shape[0],1))
    for i in range(train_set1.shape[0]):
        train_dtws[i,0] = dtw(train_set1[i,0:23], train_set1[i,23:])**2
        
    valid_dtws = numpy.zeros((valid_set1.shape[0],1))
    for i in range(valid_set1.shape[0]):
        valid_dtws[i,0] = dtw(valid_set1[i,0:23], valid_set1[i,23:])**2

    test_dtws = numpy.zeros((test_set1.shape[0],1))
    for i in range(test_set1.shape[0]):
        test_dtws[i,0] = dtw(test_set1[i,0:23], test_set1[i,23:])**2

    sess = tf.Session()
    # create two variable placehold, x for the training features, 
    # y for the labels(in this model it is the dtw distance between two time series)
    x = tf.placeholder(tf.float32, shape=[None, train_set1.shape[1]])
    y = tf.placeholder(tf.float32, shape=[None, 1])

    nn1 = NeuNet(
        None,
        None,
        input = x,
        activation = tf.nn.sigmoid,
        n_visible = train_set1.shape[1],
        n_hidden =  n_hiddens[0]
    )

    # compute the cost and minimize it
    cost = nn1.cost_function(y)
    train_step = tf.train.AdamOptimizer(learning_rate[0]).minimize(cost)

    sess.run(tf.initialize_all_variables())

    # run the model
    train_error = []
    valid_error = []
    best_valid_error = numpy.inf
    for i in range(training_epochs):
        sess.run([train_step], feed_dict={x:train_set1, y:train_dtws})
        if i%100 == 0:
            train_err = sess.run([cost],feed_dict={x:train_set1, y:train_dtws})
            train_error.append(train_err)
            valid_err = sess.run([cost],feed_dict={x:valid_set1, y:valid_dtws})
            valid_error.append(valid_err)
            print("step %d, the mean error of the training data %g, vilidation data %g"%(i, train_error[-1][0], valid_error[-1][0]))
            #print h_conv1_flat.eval(feed_dict={x:test_set1})
            if valid_error[-1][0] < best_valid_error * 0.995:
                W_best = sess.run(nn1.W)
                b_best = sess.run(nn1.b)
                best_valid_error = valid_error[-1][0]

if __name__ == '__main__':
    run_model()




