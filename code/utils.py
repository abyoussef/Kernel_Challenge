## Import essential libraries:
import pandas as pd
import numpy as np

path_to_data = '../data/'
path_to_results = '../results/'

## dictionary of the classes to identify
CLASSES = { 0:'Plane', 1:'Car', 2:'Bird', 3:'Cat',4:'Deer', 5:'Dog',6:'Frog',7:'Horse',8:'Boat',9:'Lorry'}


def load_data():
    print "Loading the data ..."
    # Load the data:
    Xtr = pd.read_csv(path_to_data + 'Xtr.csv', header=None)
    Xte = pd.read_csv(path_to_data + 'Xte.csv', header=None)
    Ytr = pd.read_csv(path_to_data + 'Ytr.csv', header=0)

    ## Transform the data to arrays
    ## and delete last column which is empty
    x_train = Xtr.drop(Xtr.columns[[3072]], 1).values
    x_test = Xte.drop(Xte.columns[[3072]], 1).values
    y_train = np.array(Ytr[[1]].values).ravel()

    print "Statistics: "
    print "There is {0} training images of dimension {1} ".format( x_train.shape[0] , x_train.shape[1])
    print "There is {0} testing  images of dimension {1} ".format( x_test.shape[0] , x_test.shape[1] )
    print "Number of classes: {0}".format(len(set(y_train)))
    return x_train, x_test, y_train

def print_info(y_train):
    ## For sanity check :
    print "Info on the classes of the images: "
    for i in range(len(CLASSES)):
        print "{0} of the class {1} ({2})".format(np.sum(y_train == i), CLASSES[i], i)

def save_results(data=None,save=False,name='Yte'):
    # data is an array of prediction of the test set
    # Set save = True to output the result to name.csv in the format of submission
    output = np.zeros((2000,2))
    output[:,0] = np.arange(1,2001)
    #if data!=None :
        #assert len(data) == 2000
    output[:,1] = data
    output = np.int32(output)
    test_output = pd.DataFrame(output,columns=['Id','Prediction'])
    # Save test_output for submission
    if save :
        print "Saving Output ... "
        test_output.to_csv(path_to_results+ name+'.csv',sep=',',index=False)
        print "Output saved."
    return test_output

if __name__ == '__main__':
    ## Load data and print info :
    x_train, x_test,y_train = load_data()
    print_info(y_train)

    ## Save results in submission format :
    save_results(save=True)

