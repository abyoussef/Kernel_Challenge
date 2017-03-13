import numpy as np
from utils import load_data
import matplotlib.pyplot as plt
path_to_save_img = '../report/img/'

def inv_preproc(l) :
    # Function to rescale array in a certain channel
    x = l - np.min(l)
    x_ = x / np.sqrt(np.var(x))
    x__ = x_ / (np.max(x_) - np.min(x_))
    x_im = x__ * 254 +1
    return x_im

## Image to matrix to plot
def img_to_rgb_matrix( l ):
    # Function to return rgb tensor of dim (32*32*3) of an image stored in array l
    size_img  = 32 * 32
    image_shape = (32,32)
    rgb_shape = (32,32,3)
    rgb_array = np.zeros(rgb_shape)
    for i in range(3):
        x = l[i*size_img :(i+1)*size_img]
        # Each channel is rescaled
        channel = inv_preproc(x).reshape(image_shape)
        rgb_array[:,:,i] = np.array(channel,dtype=int)
    return rgb_array

## Image to matrix to extract hog features
def img_to_matrix( l ):
    # Function to return rgb tensor of dim (32*32*3) of an image stored in array l
    size_img  = 32 * 32
    image_shape = (32,32)
    rgb_shape = (32,32,3)
    res = np.zeros(rgb_shape)
    for i in range(3):
        x = l[i*size_img :(i+1)*size_img]
        # Each channel is rescaled
        channel = x.reshape(image_shape)
        res[:,:,i] = np.array(channel)
    return res

def plot_img(l , save = False , name = None) :
    # Function to plot an image given its rgb array
    rgb_array = img_to_rgb_matrix(l)
    plt.imshow(rgb_array)
    if save :
        print "Saving image ... "
        assert (name != None)
        plt.savefig(path_to_save_img+name+'.png')
        print "Image saved."
    plt.show()
    return

def plot_img_train(x_train , image = None, cla = 0 ,save = False, name = None):
    if image == None :
        ## plot random image from the training set belonging to the class cla
        ind_cl = np.argwhere(y_train == cla).ravel()
        i = np.random.randint(len(ind_cl))
        print "Plot image {0} form training set".format(ind_cl[i])
        plot_img(x_train[ind_cl[i]] ,save, name);
    else :
        assert image in range(1, x_train.shape[0]+ 1  ) # We consider images from 1 to 5000 in train
        print "Plot image {0} form training set".format(image)
        plot_img(x_train[image - 1] ,save, name);
    return

def plot_img_test(x_test , image = None ,save = False, name = None):
    if image == None :
        ## plot random image from the test set
        i = np.random.randint(x_test.shape[0] + 1 )
        print "Plot image {0} form test set".format(i)
        plot_img(x_test[i] ,save, name);
    else :
        assert image in range(1, x_test.shape[0]+ 1  ) # We consider images from 1 to 2000 in test
        print "Plot image {0} form test set".format(image)
        plot_img(x_test[image - 1],save, name);
    return


if __name__ == '__main__':
    x_train, x_test,y_train = load_data()
    # Plot images in the training set : 3767
    # plot_img_train(x_train,cla = 1, save=True, name = 'img_train2' ) ## Image of a car from train set
    #plot_img_train(x_train,image = 3767 , save=True, name = 'img_train2' ) ## Image of a chosen cat from train set
    #plot_img_train(x_train,cla = 7 , save=True, name = 'img_train3' ) ## Image of a Horse from train set
    plot_img_test(x_test)
