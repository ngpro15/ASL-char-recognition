

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


# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature


# feature-descriptor-3: Haralick features
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
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

    fixed_size = tuple((200, 200))
    train_path="dataset"
    train_labels = os.listdir(train_path)
    print(train_labels)
    bins=8
    global_features = []
    labels = []
    i, j = 0, 0
    k = 0

    for training_name in train_labels:
        images_path = os.path.join(train_path, training_name)
        current_label = training_name
        k = 1
        files = [os.path.join(images_path, p) for p in sorted(os.listdir(images_path))]
        for f in files:
            print ('Extracting features from image %s' % f)
            name = f.split('/')[-1].lower()
            image = cv2.imread(f)
            image = cv2.resize(image, fixed_size)
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick   = fd_haralick(image)
            fv_histogram  = fd_histogram(image)
            result = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            labels.append(current_label)
            global_features.append(result)
            i += 1
            k += 1

        print ("[STATUS] processed folder: {}".format(current_label))
        j += 1
    print ("[STATUS] completed Global Feature Extraction...")
    
    print ("[STATUS] feature vector size {}".format(np.array(global_features).shape))
    print ("[STATUS] training Labels {}".format(np.array(labels).shape))

    # encode the target labels
    targetNames = np.unique(labels)
    le = LabelEncoder()
    target = le.fit_transform(labels)
    print ("[STATUS] training labels encoded...")

    # normalize the feature vector in the range (0-1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    print ("[STATUS] feature vector normalized...")
    print ("[STATUS] target labels: {}".format(target))
    print ("[STATUS] target labels shape: {}".format(target.shape))

    # save the feature vector using HDF5
    h5f_data = h5py.File('outputdata.h5', 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))
    h5f_label = h5py.File('outputlabels.h5', 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))
    h5f_data.close()
    h5f_label.close()
    print ("[STATUS] end of training..")
