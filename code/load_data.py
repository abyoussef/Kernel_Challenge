import pandas as pd
import numpy as np

path_to_data = '../data'
def load_data():
    Xtr = pd.read_csv(path_to_data + 'Xtr.csv' , header = None )
    Xte = pd.read_csv(path_to_data + 'Xte.csv' , header = None )
    Ytr = pd.read_csv(path_to_data + 'Ytr.csv' , header = 0 )


    ## Delete unnecessary
    X_Train = Xtr.drop(Xtr.columns[[3072]] , 1).values
    X_Test =  Xte.drop(Xte.columns[[3072]] , 1).values
    Y_Train = np.array(Ytr[[1]].values).ravel()

    return X_Train, X_Test , Y_Train
