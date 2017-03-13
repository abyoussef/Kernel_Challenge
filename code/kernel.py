## Storing Kernels :
import numpy as np


def lin_ker(x, y):
    print "Computing linear Kernel"
    return np.dot(x,y.T)

def gauss_ker(x, y, sigma =0.2):
    # Computing the gaussian kernel
    print "Computing RBF Kernel"
    z = d(x, y)
    return np.exp(-z**2/(2*(sigma**2)))

def d(x, y):
    ## Compute the matrix distance between two matrices of features
    inner_prod_matrix = np.dot(x, y.T)
    x_norm = np.sum(np.power(np.absolute(x), 2), axis=1, keepdims=True)
    y_norm = np.sum(np.power(np.absolute(y), 2), axis=1, keepdims=True)
    d_mat = x_norm + y_norm.T - 2 * inner_prod_matrix
    d_mat = np.sqrt(d_mat)
    return d_mat
