import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers.core import Dropout

# Load the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print(x_train.shape)
print(x_test.shape)

# Define a convolutional Autoencoder
latent_dim = 64 * 2


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # input layer
        self.latent_dim = latent_dim
        # 1st dense layer
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),

        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)

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

# save and load a mode
autoencoder.save('./model/')
autoencoder = keras.models.load_model('./model/')

# calculate loss and threshold
encoded_imgs = autoencoder.encoder(x_train).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
loss = tf.keras.losses.mse(decoded_imgs, x_train)
threshold = np.mean(loss) + np.std(loss)
print("Loss Threshold: ", threshold)

# Display original and reconstruction
n = 6
plt.figure(figsize=(20, 8))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

# individual sample
# Load an image from a file
img = cv2.imread('c:\\data\\test.jpg', 0)
plt.imshow(img, cmap='gray')

# nomalize img
img = img.astype('float32') / 255.
# reshape img
img = img.reshape(-1, 28, 28)

# test an image
encoded = autoencoder.encoder(img).numpy()
# plt.imshow(encoded.reshape(8, 8))
decoded = autoencoder.decoder(encoded).numpy()
# plt.imshow(decoded.reshape(28, 28))

loss = tf.keras.losses.mse(decoded, img)
print(f"Loss: {np.mean(loss) + np.std(loss)}")


small_size = 28*28 * 1
real_data_size = 851*481*3
print(f'Real data is {int(real_data_size / small_size)} times bigger')
