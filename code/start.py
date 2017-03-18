## Script to run the code :

## Import essential libraries:
from utils import load_data , save_results
from svm import ova_svm, predict_svm
from kernel import lin , rbf
from hog_feat import hog_feat_img
import numpy as np


## Upload the data:  
x_train,x_test,y_train = load_data()

# Select all the data set
N_train =  x_train.shape[0]
N_test  = x_test.shape[0]


## Dimension of the features
d = 576

X_tr = np.zeros((N_train,d))
X_te = np.zeros((N_test,d))
## Extract HOG features:
# For Training set :
print "Extracting HOG Features for training set ... "
for i in range(N_train) :
    if i % 500 == 0 :
        print "HOG features extraction for {0} images".format(i)
    X_tr[i]  = hog_feat_img(x_train[i])

print "Extraction Completed for training set ! "
print "Shape of the training features : {0}".format(X_tr.shape)
# For test set :
print "Extracting HOG Features for test set ... "
for i in range(N_test) :
    if i % 500 == 0 :
        print "HOG features extraction for {0} images ..".format(i)
    X_te[i]  = hog_feat_img(x_test[i])
print "Extraction Completed for test set ! "
print "Shape of the test features : {0}".format(X_te.shape)
## Choose Kernel (lin or kernel) :
## Training SVM Kernel One vs All for each class
alpha , bias = ova_svm(X_tr,y_train[:N_train], lambda_= 0.01,Kernel=lin)

## Prediction of the test set
y_test = predict_svm( alpha , bias, X_tr, X_te ,Kernel=lin)

## Output prediction
save_results(data=y_test,save=True,name='Yte')