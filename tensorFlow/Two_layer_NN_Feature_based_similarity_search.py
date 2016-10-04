"""
This file contained the feature-based similarity search method using two-layer neural network

Created by Zexi Chen(zchen22)
Date: Oct 2, 2016
"""

import numpy
import six.moves.cPickle as pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math

from Feature_based_similarity_search import loadData, dtw, euclideanDist, NeuNet

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

    nn2 = NeuNet(
    	None,
    	None,
    	input = nn1.output,
    	activation = tf.nn.sigmoid,
    	n_visible = n_hiddens[0]*2,
    	n_hidden = n_hiddens[1]
    )

    # compute the cost and minimize it
    cost = nn2.cost_function(y)
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
            if valid_error[-1] < best_valid_error * 0.995:
                W_best = sess.run(W_conv1)
                b_best = sess.run(b_conv1)

if __name__ == '__main__':
    run_model(n_hiddens = [100,10])
