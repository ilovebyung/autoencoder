'''
1. Extract spectrograms from wav files
2. Load training images
3. Build autoencoder 
4. Make an inference
5. [Web Interface] Copy wav files from source to target data  
6. [Web Interface] Select a day and make inference  
'''

from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import load_training_images as load_data
import extract_spectrogram as spectrogram
import autoencoder


if __name__ == "__main__":

    '''
    1. Extract spectrograms from wav files
    '''
    SOURCE = "C:/data/in"
    TARGET = "C:/data/out"
    # FIG_SIZE = (40, 40)
    FIG_SIZE = (20, 20)

    extractor = spectrogram.SpectrogramExtractor()
    extractor.extract(SOURCE, TARGET, FIG_SIZE)

    '''
    2. Load training images
    '''
    DATADIR = "C:/data/x_train"
    x_train = load_data.create_training_data(DATADIR)

    DATADIR = "C:/data/x_test"
    x_test = load_data.create_training_data(DATADIR)

    '''
    3. Build autoencoder 
    '''
    autoencoder = autoencoder.Autoencoder(latent_dim=64 * 2)
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    history = autoencoder.fit(x_train, x_train,
                              epochs=10,
                              shuffle=True,
                              validation_data=(x_test, x_test))

    # a summary of architecture
    autoencoder.encoder.summary()
    autoencoder.decoder.summary()

    # plot history
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()
