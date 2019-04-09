
import h5py
import numpy as np
import os
import glob
import cv2
from matplotlib import pyplot
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import warnings
warnings.filterwarnings('ignore')


if __name__ == "__main__":

    results = []
    scoring = "accuracy"
    num_trees = 100
    seed=9

    # import the feature vector and trained labels
    h5f_data = h5py.File('outputdata.h5', 'r')
    h5f_label = h5py.File('outputlabels.h5', 'r')
    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']
    global_features = np.array(global_features_string)
    global_labels = np.array(global_labels_string)
    h5f_data.close()
    h5f_label.close()

    print ("[STATUS] features shape: {}".format(global_features.shape))
    print ("[STATUS] labels shape: {}".format(global_labels.shape))

    print ("[STATUS] training started...")

    #training code

    # create the model - Random Forests
    model= RandomForestClassifier(n_estimators=num_trees, random_state=9)

    # fit the training data to the model
    model.fit(global_features, global_labels)
    print ("[STATUS] training completed!!!...")

    pickled_db_path="trained.pck"
    with open(pickled_db_path, 'wb') as file:
        pickle.dump(model, file)