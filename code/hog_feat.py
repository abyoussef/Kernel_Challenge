import numpy as np
from utils import load_data
from img_scale import img_to_matrix , plot_img
import matplotlib.pyplot as plt
path_to_save_img = '../report/img/'

## function to convolve the image  with an operator matrix in 2d dimension
def conv_2d(img, mat):
    c = int(mat.shape[0] / 2)
    conv_img = np.array(img, dtype=complex)

    ## Loop over each block 8 * 8
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            
            mat_sub = mat[max(0, c - i):mat.shape[0] - max(0, (i + c + 1) - img.shape[0]),
                                    max(0, c - j):mat.shape[1] - max(0, (j + c + 1) - img.shape[1])]
            img_sub = img[max(0, i - c):min(img.shape[0], i + c + 1),
                                    max(0, j - c):min(img.shape[1], j + c + 1)]
            conv_img[i, j] = np.sum(np.multiply(mat_sub, img_sub))

    return conv_img

## extract hog features for one channel
def extract_hog_feat_channel(channel):
    # Convolution matrix Gx + j Gy
    # Sobel Operator for computing the gradient 3* 3 in complex form
    mat = np.array([[-3 - 3j, 0 - 10j, +3 - 3j],[-10 + 0j, 0 + 0j, +10 + 0j], [-3 + 3j, 0 + 10j, +3 + 3j]])

    grad = conv_2d(channel, mat)
    hist = np.zeros((8, 8, 9))
    bsize = 4 # Block Size in the detection window
    nbins = 9 # Number of bins in the HOG histogram
    # Computing histogram per block for 8*8 blocks
    for i in range(8):
        for j in range(8):
            for grad_pixel in np.nditer(grad[i * bsize:(i + 1) * bsize, j * bsize:(j + 1) * bsize]):
                # Add the magnitude to the correspondant bins in the histogram
                hist[i, j, int(abs (nbins * ((360 + np.angle(case, deg=True)) % 360) / 360))] += np.absolute(grad_pixel)

    return hist, np.absolute(grad)

## Compute hog features for an image by taking the histogram of the channel with the highest gradient intensity in each block
def extract_hog_feat_img(img ) :
    img_mat = img_to_matrix(img)
    hist_rgb = np.zeros((8,8,9,3))
    hist = np.zeros((8,8,9))
    grad_rgb = np.zeros((32, 32, 3))
    grad = np.zeros((32,32))
    bsize = 4
    for i in range(3) :
        hist_rgb[:,:,:,i], grad_rgb[:,:,i] = extract_hog_feat_channel(img_mat[:,:,i])
    for j in range(8):
        for k in range(8):
            m = np.argmax(np.sum(grad_rgb[j*bsize:(j+1)*bsize ,k*bsize:(k+1)*bsize,:],axis=(0,1) ))
            hist[j,k,:] = hist_rgb[ j , k , : , m]
            grad[j * bsize:(j + 1) * bsize,k * bsize:(k + 1) * bsize ] =grad_rgb[j * bsize:(j + 1)
                                                               * bsize, k * bsize:(k + 1) * bsize, m]
    return hist,grad

# Extract HOG features in dimension 8*8*9
def hog_feat_img(img) :
    hist,_ = extract_hog_feat_img(img)
    hist_ = np.array(hist)
    return hist_.flatten()


## Function to plot gradient magnitude
def plot_grad_magn(grad,save = False , name = None):
    print "Plot Gradient Magnitude"
    plt.imshow(grad)
    if save :
        print "Saving Gradient Magnitude .. "
        assert name != None
        plt.savefig(path_to_save_img+name+'.png')
        print "Gradient Magnitude saved !"
    plt.show()


if __name__ == '__main__':
    ## Load data and print info :
    x_train, x_test,y_train = load_data()
    hist,grad = extract_hog_feat_img(x_train[789])
    plot_img(x_train[789],save = True, name = 'hog_train1')
    plot_grad_magn(grad,save = True, name = 'grad_train1')
    hist1, grad1 = extract_hog_feat_img(x_train[2384])
    plot_img(x_train[2384], save=True, name='hog_train2')
    plot_grad_magn(grad1, save=True, name='grad_train2')