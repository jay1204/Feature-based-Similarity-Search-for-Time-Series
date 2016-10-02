"""
Model a stacked denoising autoencoder with multiple layers with real-value inputs or binary inputs
The code is based on the theano digit recognition tutorial
Created: Zexi Chen(zchen22)
Date: Aug 20
"""

import os 
import sys
import numpy
import timeit

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import six.moves.cPickle as pickle

from load_data_Hie_knn import load_data
from dA_Hie_knn import dA
import matplotlib.pyplot as plt

class SdA(object):
    """
    stacked denoising auto-encoder(SdA)

    The hiddenlyaer of the dA at layer 'i' becomes the input of the dA at layer 'i + 1'
    The first layer dA gets the input from real input data and the hidden layer of the last 
    dA represents the reconstructed real input data. After pretrain the model layer by layer,
    the final model will be obtained by tuning all the parameters together
    """

    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=23,
        hidden_layers_size=[128,32],
        corruption_levels=[0.0,0.0],
        v_h_learning_rates=[0.1,0.1],
        h_v_learning_rates=[0.1,0.1]
    ):
        #self.hidden_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_size)
        self.v_h_learning_rates = v_h_learning_rates
        self.h_v_learning_rates = h_v_learning_rates
        self.corruption_levels = corruption_levels

        assert self.n_layers > 0

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        self.x = T.matrix('x') 

        # The SdA is an MLP, for which all weights of intermediate layers
        # are shared with a different denoising autoencoders
        # We will first construct the SdA as a deep multilayer perceptron,
        # and when constructing each sigmoidal layer we also construct a
        # denoising autoencoder that shares weights with that layer
        # During pretraining we will train these autoencoders (which will
        # lead to chainging the weights of the MLP as well)
        # During finetunining we will finish training the SdA by doing
        # stochastich gradient descent on the MLP

        for i in range(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_size[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.dA_layers[-1].output

            '''
            hidden_layer = HiddenLayer(rng=numpy_rng,
                                        input = layer_input,
                                        n_in = input_size,
                                        n_out=hidden_layers_size[i],
                                        activation=T.nnet.sigmoid)

            self.hidden_layers.append(hidden_layer)
            '''

            if i == 0:
                dA_layer = dA(numpy_rng=numpy_rng,
                              theano_rng = theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_size[i],#W=hidden_layer.W,bhid=hidden_layer.b,
                              v_h_active = T.nnet.sigmoid)
            else:
                dA_layer = dA(numpy_rng=numpy_rng,
                              theano_rng = theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_size[i],#W=hidden_layer.W,bhid=hidden_layer.b,
                              v_h_active = T.nnet.sigmoid,
                              h_v_active = T.nnet.sigmoid
                              )
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)


        #reconstructed_input = self.x
        #reconstructed_input = self.dA_layers[0].x
        #for i in range(self.n_layers):
        #    reconstructed_input = self.dA_layers[i].get_hidden_values(reconstructed_input)
        #reconstructed_input = [self.x]
        #for i in range(self.n_layers):
        #    temp = self.dA_layers[i].get_hidden_values(reconstructed_input[-1])
        #    reconstructed_input.append(temp)
        reconstructed_input = []
        reconstructed_input.append(self.dA_layers[-1].output)
        for i in range(self.n_layers - 1, -1, -1):
            temp = self.dA_layers[i].get_reconstructed_input(reconstructed_input[-1])
            reconstructed_input.append(temp)

        self.finetune_cost = self.dA_layers[0].get_error(reconstructed_input[-1])

    def pretraining_function(self, train_set, batch_size):
        """
        Generate a list of functions, each of them implementing one step in training
        the dA corresponding to the layer with same index. The function will require
        as input the minibatch index, and to train a dA, you just need to iterate the 
        corresponding function on all minibatch indexes.
        """
        index = T.lscalar('index') # index of a minibatch

        batch_begin = index * batch_size
        batch_end = batch_begin +  batch_size

        pretrain_fns = []
        for i, dA in enumerate(self.dA_layers):
            cost, updates = dA.get_cost_updates(corruption_level = self.corruption_levels[i],
                                v_h_learning_rate = self.v_h_learning_rates[i],
                                h_v_learning_rate = self.h_v_learning_rates[i]
                                )

            fn = theano.function(
                inputs = [index],
                outputs = cost,
                updates = updates,
                givens = {
                    self.x:train_set[batch_begin: batch_end]
                }
            )

            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, dataset, batch_size, finetune_v_h_learning_rates, finetune_h_v_learning_rates):
        """
            Generate a function that implements one step of the finetuning
        """
        datasets = load_data(dataset)
        train_set = datasets[0]
        valid_set = datasets[1]
        test_set = datasets[2]

        # compute the number of minibatches for training, validation and testing
        n_valid_batches = valid_set.get_value(borrow=True).shape[0] // batch_size
        n_test_batches = test_set.get_value(borrow=True).shape[0] // batch_size

        index = T.lscalar('index') # the indice of the mini-batch
        print len(self.dA_layers[0].params), len(self.dA_layers[1].params), len(self.params)
        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)
        print gparams        

        # compute the updates 
        updates = []
        list_index = 0
        for i in range(self.n_layers):
            if not self.dA_layers[i].tied:
                gaps =  4
            else:
                gaps =  3
            for j in range(gaps):
                if j >= 0 and j <= 1:
                    updates.append((self.params[list_index + j], self.params[list_index + j] - gparams[list_index + j] * finetune_v_h_learning_rates[i]))
                else:
                    updates.append((self.params[list_index + j], self.params[list_index + j] - gparams[list_index + j] * finetune_h_v_learning_rates[i]))
            list_index += gaps


        train_fn = theano.function(
            inputs = [index],
            outputs = self.finetune_cost,
            updates = updates,
            givens = {
                self.x: train_set[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name = 'train'
        )

        test_fn = theano.function(
            inputs = [index],
            outputs = self.finetune_cost,
            givens = {
                self.x: test_set[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name = 'test'
        )

        valid_fn = theano.function(
            inputs = [index],
            outputs = self.finetune_cost,
            givens = {
                self.x: valid_set[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name = 'valid'
        )

        # create a function that scans the entire validation set
        def valid_score():
            return [valid_fn(i) for i in range(n_valid_batches)]

        def test_score():
            return [test_fn(i) for i in range(n_test_batches)]

        return train_fn, valid_score, test_score

def test_SdA(v_h_learning_rates, h_v_learning_rates,finetune_v_h_learning_rates,
            finetune_h_v_learning_rates, corruption_levels, hidden_layers_size = [128,32],
             pre_training_epochs=50, training_epochs=500, dataset='samples',
             batch_size=100):
    
    datasets = load_data(dataset)

    train_set = datasets[0]
    valid_set = datasets[1]
    test_set = datasets[2]

    n_train_batches = train_set.get_value(borrow=True).shape[0] // batch_size

    numpy_rng = numpy.random.RandomState(111)
    print('...building the model')
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng = numpy_rng,
        n_ins = 23,
        hidden_layers_size = hidden_layers_size,
        corruption_levels = corruption_levels,
        v_h_learning_rates = v_h_learning_rates,
        h_v_learning_rates = h_v_learning_rates
    )

    print('...getting the pretraining functions')
    pretraining_fns = sda.pretraining_function(train_set=train_set, batch_size=batch_size)

    print('...pre-training the model')
    start_time = timeit.default_timer()
    for i in range(sda.n_layers):
        # pretrain the model
        for epoch in range(pre_training_epochs):
            c = []
            for batch_index  in range(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index))

            print('Pre-training layer %i, epoch %d, cost %f' % (i, epoch, numpy.mean(c)))

    end_time = timeit.default_timer()

    print('The code ran for %.2fm' % ((end_time - start_time)/60.))

    # Finetuning the model
    print('...getting the finetuning functions')
    train_fn, validate_model, test_model = sda.build_finetune_functions(dataset=dataset, batch_size=batch_size, 
                                                                        finetune_v_h_learning_rates = finetune_v_h_learning_rates, 
                                                                        finetune_h_v_learning_rates = finetune_h_v_learning_rates)

    print('...Finetuning the model')

    start_time = timeit.default_timer()
    train_errors = []
    valid_errors = []
    best_valid_error = numpy.inf
    for epoch in range(training_epochs):
        train_cost = []
        for minibatch in range(n_train_batches):
            train_cost.append(train_fn(minibatch))
        validation_losses = validate_model()
        this_validation_loss = numpy.mean(validation_losses)
        train_errors.append(numpy.mean(train_cost))
        valid_errors.append(this_validation_loss)
        print 'Training epoch %d, cost ' % epoch, train_errors[-1]
        print 'Validation cost ', valid_errors[-1]

        if valid_errors[-1] < best_valid_error * 0.995:
            best_valid_error = valid_errors[-1]

            with open('data/best_SdA_model.pkl', 'wb') as f:
                pickle.dump([param.get_value() for param in sda.params],
                            f, protocol=pickle.HIGHEST_PROTOCOL)
    """
    patience =  10 * n_train_batches
    patience_increase = 2. 
    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience // 2)

    best_validation_loss = numpy.inf
    test_score = 0.
    

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            minibatch_avg_cost = train_fn(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                validation_losses = validate_model()
                this_validation_loss = numpy.mean(validation_losses)
                print('epoch %i, minibatch %i/%i, validation error %f %%' % 
                        (epoch, minibatch_index + 1, n_train_batches,
                            this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:

                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # apply the model to the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('    epoch %i, minibatch %i/%i, test error of '
                            'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))
            if patience <= iter:
                done_looping = True
                break

    print(
        (
            'Optimization complete with best validation score of %f %%, '
            'on iteration %i, '
            'with test performance %f %%'
        )
        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
    )
    """
    end_time = timeit.default_timer()
    print('The fine tuning code ran for %.2fm' % ((end_time - start_time)/60.))
    Xaxis = range(training_epochs)
    plt.plot(Xaxis, train_errors, 'r',label='train error')
    plt.plot(Xaxis, valid_errors, 'g',label='validation error')
    plt.show()


if __name__ == '__main__':
    test_SdA(v_h_learning_rates = [0.00001,0.1, 0.1], h_v_learning_rates=[0.0001,0.1,0.1], 
            finetune_v_h_learning_rates = [0.0000001, 0.0001, 0.0001], 
            finetune_h_v_learning_rates = [0.0000001, 0.0001, 0.0001],
            corruption_levels=[0.05,0.0,0.0],
            hidden_layers_size = [128,32,16],pre_training_epochs=1000, training_epochs=1000)

