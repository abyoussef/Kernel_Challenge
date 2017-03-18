import numpy as np
import cvxopt

# SVM file DONE !

def ova_svm( X , y, lambda_, Kernel):
    # One vs All SVM implementation
    n = X.shape[0]  # is the
    num_labels = len(set(y)) # K is the number of classes

    # Parameters estimated by the training SVM for each label
    a = np.zeros((num_labels, n))
    b = np.zeros(num_labels) # biases for each label

    # Kernel matrix
    K = Kernel(X,X) # Kernel Matrix of size n*n
    # For each label, we perform an SVM One vs All classification
    print "Start one vs all classification"
    for k  in range( num_labels ):
        ova_y = 2 * (y==k) - 1
        a[k, :], b[k] = solve_dual_svm(K, ova_y , lambda_)
        print "Classification completed for label {0} .. ".format(k)
    print "One Vs All Classification Completed !"
    return a, b



def solve_dual_svm(K, y, lambda_):
    ## Function to solve the dual formulation of the SVM
    ## Based on the http://cvxopt.org/examples/tutorial/qp.html?highlight=quadratic

    n = K.shape[0]
    gamma = 1 / (2 * lambda_ * n)
    P = cvxopt.matrix(K)
    h = cvxopt.matrix(0., (2 * n, 1))
    h[:n] = gamma
    A = cvxopt.matrix(1., (1, n))
    b = cvxopt.matrix(0.)
    y = y.astype(np.double)
    y_diag = cvxopt.spdiag(y.tolist())
    q = cvxopt.matrix(-y)
    G = cvxopt.sparse([y_diag, - y_diag])
    res = cvxopt.solvers.qp(P, q, G, h, A, b)

    return np.array(res["x"]).T, res["y"][0]



def predict_svm( a , b, X_tr_features, X_te_features ,Kernel):
    # Predict labels of the X_te_features based on the solution of the SVM
    prob_pred = np.zeros((a.shape[0],X_te_features.shape[0])) # Matrix of size num_labels * test size
    print "Start prediction ... "
    K = Kernel(X_te_features,X_tr_features)
    for k in range(a.shape[0]):
        prob_pred[k,:] = np.dot(K, a[k,:]) + b[k]
    print "Prediction completed !"
    return np.argmax(prob_pred, axis=0)


