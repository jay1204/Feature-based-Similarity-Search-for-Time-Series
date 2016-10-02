import numpy
import six.moves.cPickle as pickle
import theano

fileObject = open('data/best_SdA_model.pkl','r')
data_set = pickle.load(fileObject)

params = [numpy.array(dataset) for dataset in data_set]

fileObject1 = open('data/samples','r')
samples = pickle.load(fileObject1)

time_series = numpy.array(samples)