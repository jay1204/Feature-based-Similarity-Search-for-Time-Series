# Feature-based-Similarity-Search-for-Time-Series
###Develop a novel method for fast retrieval of k-Nearest Neighbors for time series curves

#####The theano folder constains the files that apply Stacked Denoising Autoencoder Method to our time series data for dimensionality reduction. 

#####The tensorflow folder contains the files that apply fully connected multiple layers neural network to our time series data, with the cost function defined as the square error of the difference between squared Dynamic Time Warping distance in original space and squared Euclidean distance in new feature space for any two time series.
1. The file "Feature_based_similarity_search.py" contains the Neural Network class, and a one layer neural network model
2. The file "Two_layer_NN_Feature_based_similarity_search.py" contains a two-layer Neural Network model.
