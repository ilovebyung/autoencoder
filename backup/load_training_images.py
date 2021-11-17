import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

'''
Read jpg files, and create training data
X image,y label
'''


def create_training_data(DATADIR):
    training_data = []
    # for category in CATEGORIES:  # "baseline" and "rattle"

    #     path = os.path.join(DATADIR, category)  # create path
    #     # get the classification  (0 or a 1). 0=baseline 1=rattle
    #     class_index = CATEGORIES.index(category)

    # iterate over each image
    for image in os.listdir(DATADIR):
        try:
            data = cv2.imread(os.path.join(DATADIR, image))
            # resize to make sure data consistency
            resized_data = cv2.resize(data, (2174, 2232))
            # add this to our training_data
            training_data.append([resized_data])
        except Exception as err:
            print("an error has occured: ", err, os.path.join(DATADIR, image))

    # normalize data
    X = np.array(training_data)/255.
    # reshape
    X = np.array(X).reshape(-1, 2232, 2174, 3)
    return X


# DATADIR = "C:/data/out"
# training_data = []
# images = os.listdir(DATADIR)
# image = images[0]
# data = cv2.imread(os.path.join(DATADIR, image))
# # resize to make sure data consistency
# resized_data = cv2.resize(data, (2174, 2232))
# # add this to our training_data
# training_data.append([resized_data])

# # normalize data
# X = np.array(training_data)/255.
# # reshape
# X = np.array(X).reshape(-1, 2232, 2174, 3)


if __name__ == "__main__":

    '''
    Check the size of an image
    '''
    file = 'C:/data/out/2118611119H001004072_TDM_2021-08-27_17-01-04__Microphone.jpg'
    image = plt.imread(file)
    height, length, _ = image.shape

    '''
    Loading data and create training_data
    '''

    # DATADIR = "C:/data/spectrogram"
    DATADIR = "C:/data/out"
    # CATEGORIES = ["baseline", "rattle"]

    X = create_training_data(DATADIR)

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
