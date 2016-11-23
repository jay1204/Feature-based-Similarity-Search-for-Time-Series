import numpy
import six.moves.cPickle as pickle
import math

def loadData(path):
    fileObject = open(path, 'r')
    dataset = pickle.load(fileObject)

    return dataset

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

def euclideanDist(list1,list2):
    distance = 0
    for x in range(len(list1)):
        distance += pow((list1[x]-list2[x]),2)
    return math.sqrt(distance)