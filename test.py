import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import scipy
import pickle
import random
import mahotas
from scipy.misc import imread
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import h5py
import glob
import imutils
import warnings
warnings.filterwarnings('ignore')


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(image).mean(axis=0)
    # return the result
    return haralick


# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()



if __name__ == "__main__":
    bins=8
    with open("trained.pck", 'rb') as file:  
        pickle_model = pickle.load(file)
    train_path= "dataset"
    train_labels = os.listdir(train_path)
    fixed_size = tuple((200, 200))

    test_path= "tester"

    files = [os.path.join(test_path, p) for p in sorted(os.listdir(test_path))]
    for f in files:
        print ('Extracting features from image %s' % f)
        name = f.split('/')[-1].lower()
        image = cv2.imread(f)
        image = cv2.resize(image, fixed_size)
        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
 
        result = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
        prediction = pickle_model.predict(result.reshape(1,-1))[0]


        cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

        # display the output image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.show()
