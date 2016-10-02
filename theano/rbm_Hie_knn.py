"""
	This file is to build Gaussian RBM
	Created by: Zexi Chen(zchen22)
	Date: Aug 10
"""

import os 
import sys
import numpy
import timeit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from load_data_Hie_knn import load_data
