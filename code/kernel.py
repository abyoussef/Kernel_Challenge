## Storing Kernels :
import numpy as np


def lin(x, y):
    print "Computing linear Kernel"
    return np.dot(x,y.T)
# Defining Gaussian Kernel 
def rbf(x, y, sigma= 0.2):
    print "Computing RBF Kernel"
    z = d(x, y)
    return np.exp(-z**2/(2*(sigma**2)))

def d(x, y):
    ## Compute the matrix distance between two matrices of features
    gram_matrix = np.dot(x, y.T)

    x_norm = np.sum( np.abs(x) ** 2 , axis=1, keepdims=True)
    y_norm = np.sum( np.abs(y) ** 2 , axis=1, keepdims=True)

    d_mat = x_norm + y_norm.T - 2 * gram_matrix

    d_mat= np.sqrt(np.maximum(d_mat,0))

    return d_mat