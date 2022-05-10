'''
1. Extract spectrograms from wav files
2. Load training images
3. Build autoencoder 
4. Set threshold
5. Make an inference
'''

from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from tensorflow.keras.layers import Dense, Input, Conv2D, Flatten, Reshape, Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score
import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pathlib
import librosa
import librosa.display
import concurrent.futures

'''
Read wav files from SOURCE folder, extract spectrograms in JPG format, and save in TARGET folder
'''

'''
1. Extract spectrograms from wav files
'''


class SpectrogramExtractor:
    def extract(self, SOURCE, TARGET, FIG_SIZE):
        os.chdir(SOURCE)
        for file in os.listdir(SOURCE):
            # check file extention
            if file.endswith(".wav"):
                # load audio file with Librosa
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(librosa.load, file, sr=22050)
                    signal, sample_rate = future.result()

                # perform Fourier transform (FFT -> power spectrum)
                fft = np.fft.fft(signal)

                # calculate abs values on complex numbers to get magnitude
                spectrum = np.abs(fft)

                # create frequency variable
                f = np.linspace(0, sample_rate, len(spectrum))

                # take half of the spectrum and frequency
                left_spectrum = spectrum[:int(len(spectrum)/2)]
                left_f = f[:int(len(spectrum)/2)]

                # STFT -> spectrogram
                hop_length = 512  # in num. of samples
                n_fft = 2048  # window in num. of samples

                # calculate duration hop length and window in seconds
                hop_length_duration = float(hop_length)/sample_rate
                n_fft_duration = float(n_fft)/sample_rate

                # perform stft
                stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

                # calculate abs values on complex numbers to get magnitude
                spectrogram = np.abs(stft)  # np.abs(stft) ** 2

                # apply logarithm to cast amplitude to Decibels
                log_spectrogram = librosa.amplitude_to_db(spectrogram)

                # Matplotlib plots: removing axis, legends and white spaces
                plt.figure(figsize=FIG_SIZE)
                plt.axis('off')
                librosa.display.specshow(
                    log_spectrogram, sr=sample_rate, hop_length=hop_length)
                data_path = pathlib.Path(TARGET)
                file_name = f'{file[0:-4]}.jpg'
                full_name = str(pathlib.Path.joinpath(data_path, file_name))
                plt.savefig(str(full_name), bbox_inches='tight', pad_inches=0)


'''
2. Load training images  
'''
# resize and normalize data for training


def create_training_data(data_path, size=224):
    training_data = []
    # for category in CATEGORIES:  # "baseline" and "rattle"

    #     path = os.path.join(data_path, category)  # create path
    #     # get the classification  (0 or a 1). 0=baseline 1=rattle
    #     class_index = CATEGORIES.index(category)

    # iterate over each image
    for image in os.listdir(data_path):
        # check file extention
        if image.endswith(".jpg"):
            try:
                data_path = pathlib.Path(data_path)
                full_name = str(pathlib.Path.joinpath(data_path, image))
                data = cv2.imread(str(full_name), 0)
                # resize to make sure data consistency
                resized_data = cv2.resize(data, (size, size))
                # add this to our training_data
                training_data.append([resized_data])
            except Exception as err:
                print("an error has occured: ", err, str(full_name))

    # normalize data
    training_data = np.array(training_data)/255.
    # reshape
    training_data = np.array(training_data).reshape(-1, size, size)
    return training_data


'''
3. Build autoencoder 
'''
# Define a convolutional Autoencoder


# class Autoencoder(Model):
#     def __init__(self, latent_dim):
#         super(Autoencoder, self).__init__()
#         # input layer
#         self.latent_dim = latent_dim
#         # 1st dense layer
#         self.encoder = tf.keras.Sequential([
#             layers.Flatten(),
#             layers.Dense(latent_dim, activation='relu'),

#         ])
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(224*224, activation='sigmoid'),
#             layers.Reshape((224, 224))
#         ])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


'''
4. Set threshold
'''


# def model_threshold(autoencoder, x_train):
#     encoded_imgs = encoder(x_train).numpy()
#     decoded_imgs = decoder(encoded_imgs).numpy()
#     loss = tf.keras.losses.mse(decoded_imgs, x_train[:, :, :, np.newaxis])
#     threshold = np.mean(loss) + np.std(loss)
#     return threshold


def spectrogram_loss(encoder, decoder, spectrogram, size=224):
    data = np.ndarray(shape=(1, size, size), dtype=np.float32)
    # individual sample
    # Load an image from a file
    data = cv2.imread(str(spectrogram), 0)
    # resize to make sure data consistency
    resized_data = cv2.resize(data, (size, size))
    # nomalize img
    normalized_data = resized_data.astype('float32') / 255.
    # test an image
    encoded = encoder(normalized_data.reshape(-1, size, size))
    decoded = decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    return sample_loss


'''
5. Make an inference
'''


def spectrogram_loss(autoencoder, spectrogram, size=224):
    data = np.ndarray(shape=(1, size, size), dtype=np.float32)
    # individual sample
    # Load an image from a file
    data = cv2.imread(str(spectrogram), 0)
    # resize to make sure data consistency
    resized_data = cv2.resize(data, (size, size))
    # nomalize img
    normalized_data = resized_data.astype('float32') / 255.
    # test an image
    encoded = autoencoder.encoder(normalized_data.reshape(-1, size, size))
    decoded = autoencoder.decoder(encoded)
    loss = tf.keras.losses.mse(decoded, normalized_data)
    sample_loss = np.mean(loss) + np.std(loss)
    return sample_loss


if __name__ == "__main__":

    '''
    1. Extract spectrograms from wav files
    '''
    SOURCE = "C:/data/in"
    TARGET = "C:/data/out"
    FIG_SIZE = (20, 20)
    args = [SOURCE, TARGET, FIG_SIZE]

    import time
    start = time.perf_counter()

    extractor = SpectrogramExtractor()
    extractor.extract(SOURCE, TARGET, FIG_SIZE)

    finish = time.perf_counter()
    print(f'Finished in {round(finish-start, 2)} second(s)')

    '''
    2. Load training images
    '''
    data_path = "C:/data/x_train"
    x_train = create_training_data(data_path)

    data_path = "C:/data/x_test"
    x_test = create_training_data(data_path)

    '''
    3. Build autoencoder 
    '''

    # network parameters
    size = 224
    input_shape = (size, size, 1)
    batch_size = 32
    kernel_size = 3
    latent_dim = 16
    # encoder/decoder number of CNN layers and filters per layer
    layer_filters = [32, 64]

    # build the autoencoder model
    # first build the encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    # stack of Conv2D(32)-Conv2D(64)
    for filters in layer_filters:
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)

    # shape info needed to build decoder model
    # so we don't do hand computation
    # the input to the decoder's first
    # Conv2DTranspose will have this shape
    # shape is (7*8, 7*8, 64*8) which is processed by
    # the decoder back to (28*8, 28*8, 1)
    shape = K.int_shape(x)
    # generate latent vector
    x = Flatten()(x)
    latent = Dense(latent_dim, name='latent_vector')(x)
    # instantiate encoder model
    encoder = Model(inputs,
                    latent,
                    name='encoder')
    encoder.summary()
    plot_model(encoder,
               to_file='encoder.png',
               show_shapes=True)

    # build the decoder model
    latent_inputs = Input(shape=(latent_dim,), name='decoder_input')
    # use the shape (7, 7, 64) that was earlier saved
    x = Dense(shape[1] * shape[2] * shape[3])(latent_inputs)
    # from vector to suitable shape for transposed conv
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    # stack of Conv2DTranspose(64)-Conv2DTranspose(32)
    for filters in layer_filters[::-1]:
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,   activation='relu',
                            strides=2,
                            padding='same')(x)
    # reconstruct the input
    outputs = Conv2DTranspose(filters=1,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='decoder.png', show_shapes=True)
    # autoencoder = encoder + decoder
    # instantiate autoencoder model
    autoencoder = Model(inputs,
                        decoder(encoder(inputs)),
                        name='autoencoder')

    autoencoder.summary()
    plot_model(autoencoder,
               to_file='autoencoder.png',
               show_shapes=True)

    # Mean Square Error (MSE) loss function, Adam optimizer
    autoencoder.compile(loss='mse', optimizer='adam')
    # train the autoencoder
    history = autoencoder.fit(x_train,
                              x_train,
                              validation_data=(x_test, x_test),
                              epochs=20,
                              batch_size=batch_size)

    # autoencoder = Autoencoder(latent_dim=64 * 2)
    # autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # history = autoencoder.fit(x_train, x_train,
    #                           epochs=10,
    #                           shuffle=True,
    #                           validation_data=(x_test, x_test))

    # autoencoder = build_autoencoder(size=224, latent_dim=64 * 2)
    # autoencoder.compile(loss='mse')
    # history = autoencoder.fit(
    #     x_train,
    #     x_train,
    #     epochs=20,
    #     batch_size=32, validation_split=0.10)

    # # a summary of architecture
    # autoencoder.encoder.summary()
    # autoencoder.decoder.summary()

    '''
    '''

    # plot history
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.legend()
    plt.show()

    # save and load a mode
    autoencoder.save('./model/')
    autoencoder = keras.models.load_model('./model/')

    # load autoencoder model
    if autoencoder is None:
        # autoencoder = Autoencoder(latent_dim=64 * 2)
        autoencoder = keras.models.load_model('./model/')

    '''
    4. Set threshold
    '''

    encoded_imgs = encoder(x_train).numpy()
    decoded_imgs = decoder(encoded_imgs).numpy()
    loss = tf.keras.losses.mse(decoded_imgs, x_train[:, :, :, np.newaxis])
    threshold = np.mean(loss) + np.std(loss)

    print("Loss Threshold: ", threshold)

    # load autoencoder model
    if autoencoder is None:
        autoencoder = keras.models.load_model('./model/')

    '''
    5. Make an inference
    '''
    # get statistics for each spectrogram
    file = 'c:/data/sample_0.jpg'
    file = 'c:/data/sample_1.jpg'
    # file = 'c:/data/sample_2.jpg'
    sample = plt.imread(file)
    plt.imshow(sample)
    sample = pathlib.Path(file)

    encoded_imgs = encoder(x_train).numpy()
    decoded_imgs = decoder(encoded_imgs).numpy()
    loss = tf.keras.losses.mse(decoded_imgs, x_train[:, :, :, np.newaxis])
    threshold = np.mean(loss) + np.std(loss)

    sample_loss = spectrogram_loss(encoder, decoder, sample)

    if sample_loss > threshold:
        print(
            f'Loss is bigger than threshold \n \
              Sample Loss: {sample_loss} \n \
              Threshold: {threshold} ')
    else:
        print(
            f'Loss is smaller than threshold \n \
              Sample Loss: {sample_loss} \n \
              Threshold: {threshold} ')
