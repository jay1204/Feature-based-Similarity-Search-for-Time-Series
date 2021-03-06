"""
The goal of this file is to build multiple MLPs to train the time series data
to re-align the distorted features in time series data. We explore DTW distance
as the training criteria

Author: Zexi Chen(zchen22)
Date: Jan 26, 2017
"""

import numpy as np
import six.moves.cPickle as pickle
import tensorflow as tf
from ConvLayer import ConvLayer

class FullyConnectedNet(object):
    """
    Build a fully-connected neural network with an arbitrary number of hidden layers,
    The regularization is implemented by dropout
    For a network of L layers, the architecture will be
    
    {affine - relu - [dropout]} * (L - 1) - affine -relu
    """ 

    def __init__(self, x, y, hidden_dims, input_dim = 23, dropout = 0, reg = 0.0, 
            weight_scale = 1e-2, dtype = np.float32):
    
        """
        Initialize a new fullyconnectednet
        
        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer
        - input_dim: An integer giving the size of the input
        - dropout: scalar between 0 and 1 giving dropout strength, if dropout==0, then
                   do not use dropout at all
        - reg: scalar giving l2 regularization strength
        - dtype: A numpy datatype object: all variables will use this type. 
        """
        self.reg = reg
        self.layers = []
        self.num_layers = len(hidden_dims)
        self.y = y
        
        for i in xrange(self.num_layers):
            if i == 0:
                layer = ConvLayer(
                        None, 
                        None, 
                        input = x,
                        weight_scale = weight_scale,
                        activation = tf.nn.relu,
                        n_visible = input_dim * 2,
                        n_hidden = hidden_dims[i],
                        dropout = dropout
                        )
            else:
                layer = ConvLayer(
                        None, 
                        None, 
                        input = self.layers[-1].output,
                        weight_scale = weight_scale,
                        activation = tf.nn.relu,
                        n_visible = hidden_dims[i - 1] * 2,
                        n_hidden = hidden_dims[i],
                        dropout = dropout
                        )
            self.layers.append(layer)
        
    def loss(self):
        loss = self.layers[-1].cost_function(self.y)
        if self.reg > 0.0:
            for layer in self.layers:
                loss += 0.5 * reg * np.sum(layer.W ** 2)
        return loss
    
