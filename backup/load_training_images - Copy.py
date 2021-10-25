import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm

'''
Read jpg files, and create training data
X image,y label
'''


def create_training_data(DATADIR, CATEGORIES):
    training_data = []
    for category in CATEGORIES:  # "baseline" and "rattle"

        path = os.path.join(DATADIR, category)  # create path
        # get the classification  (0 or a 1). 0=baseline 1=rattle
        class_index = CATEGORIES.index(category)

        # iterate over each image per dogs and cats
        for image in tqdm(os.listdir(path)):
            try:
                data = cv2.imread(os.path.join(path, image))
                # resize to make sure data consistency
                resized_data = cv2.resize(data, (1674, 815))
                # add this to our training_data
                training_data.append([resized_data, class_index])
            except Exception as err:
                print("an error has occured: ", err, os.path.join(path, image))

        # create_training_data()
        X = []
        y = []

        for features, label in training_data:
            X.append(features)
            y.append(label)

        X = np.array(X)/255.
        return X, y


if __name__ == "__main__":

    '''
    Loading data and create training_data
    '''
    # DATADIR = "C:/data/spectrogram"
    DATADIR = "C:/data/out"
    CATEGORIES = ["baseline", "rattle"]

    X, y = create_training_data(DATADIR, CATEGORIES)

    # Visualize data
    # Display original and reconstruction
    n = 4
    plt.figure(figsize=(10, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(X[i])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
