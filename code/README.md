In the code folder, you can find the code organized as follows :

Markup: * start.py : to run the classification task;
* kernel.py : Store the kernels used in the kernel SVM method;
* svm.py : solve the dual of the svm method with quadratic programming;
* img_scale.py : useful for two main reasons : 
          * rescale the array of the image in order to visualize it. We can visualize images in training set by their classes or randomly visualize images from test set.
          * Transform an array of the image into a matrix without rescaling in order to extract HOG features
* hog_feat.py : Extract HOG features from an image with convolution with Sobel operator. We can vizualize the magnitude
    of the gradient with this script. Note: HOG are computed for each RGB channel and then they are grouped by the
    maximum of the gradient magnitude with respect to each block;
* utils.py : Contains utils function to load data and output results in the format accepted by the Challenge.


* log_lin.txt : Contains log file for the compilation of start.py with linear kernel
* log_rbf.txt : Contains log file for the compilation of start.py with gaussian(rbf) kernel


In the same level as the code folder should be :

    * data : folder to store the data
    * results : to output the predictions
