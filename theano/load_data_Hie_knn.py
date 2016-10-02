"""
This file is to read and preprocess the input time series data
Created by: Zexi Chen(zchen22)
Date: Aug 10
"""

import os 
import sys
import numpy
import six.moves.cPickle as pickle
import theano

def load_data(data_path):
    data_dir, data_file = os.path.split(data_path)
    if data_dir == "" and not os.path.isfile(data_path):
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            data_path
        )
        if os.path.isfile(new_path):
            data_path = new_path

    print('......loading data')

    # read the data
    # the dataset is a numpy.ndarray of 2 dimensions
    # where each row corresponds to one example
    fileObject = open(data_path, 'r')
    data_set = pickle.load(fileObject)
    length = len(data_set)

    def shared_dataset(dataset, borrow=True):
        """
        This function is to load the dataset inot the shared variables
        The reason why we store it in shared variables is to allow
        Theano to copy it into GPU memory. Then we do not need to copy 
        it from CPU to GPU every time.
        """ 
        shared_v = theano.shared(numpy.asarray(dataset, dtype = theano.config.floatX),borrow = borrow)

        # When storing data on the GPU, it has to be stored as floats
        # But after storing the data, we can cast it to int using 
        # theano.tensor.cast(shared_data, 'int32')
        return shared_v

    # using the function to copying the data from CPU to GPU
    train_set_x = shared_dataset(data_set[:length * 8 // 10])
    valid_set_x = shared_dataset(data_set[length * 8 // 10 : length * 9 // 10])
    test_set_x = shared_dataset(data_set[length * 9 // 10 :])

    rval = [train_set_x, valid_set_x, test_set_x]
    print('Finish loading data.')

    return rval
    
