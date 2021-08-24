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

# Get train_images and test_images
# train
train = []
data = zip(x_train, y_train)
for image, label in data:
    if label == 2:
        train.append(image)

train_images = np.array(train)
train_images = train_images.astype('float32') / 255.

# test
test = []
data = zip(x_test, y_test)
for image, label in data:
    if label == 2:
        train.append(image)

test_images = np.array(test)
test_images = test_images.astype('float32') / 255.


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
            layers.Dense(latent_dim, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(latent_dim/2, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim/2, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(784, activation='sigmoid',
                         kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.Reshape((28, 28))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

history = autoencoder.fit(train_images, train_images,
                          epochs=10,
                          shuffle=True,
                          validation_data=(test_images, test_images))

# a summary of architecture
autoencoder.encoder.summary()
autoencoder.decoder.summary()

# plot history
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()

# calculate loss and threshold
encoded_imgs = autoencoder.encoder(train_images).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
loss = tf.keras.losses.mse(decoded_imgs, train_images)
threshold = np.mean(loss) + np.std(loss)
print("Threshold: ", threshold)

# Display original and reconstruction
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(train_images[i])
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
img = x_test[101, :, :]
plt.imshow(img)
img = x_test[101, :, :].reshape(1, 28, 28)
encoded = autoencoder.encoder(img).numpy()
plt.imshow(encoded.reshape(8, 8))
decoded = autoencoder.decoder(encoded).numpy()
plt.imshow(decoded.reshape(28, 28))

# Load an image from a file
img = cv2.imread('c:\\data\\test.jpg', 0)
# plt.imshow(img)

# nomalize img
img = img.astype('float32') / 255.
# reshape img
img = img.reshape(1, 28, 28)

# test an image
encoded = autoencoder.encoder(img).numpy()
# plt.imshow(encoded.reshape(8,8))
decoded = autoencoder.decoder(encoded).numpy()
# plt.imshow(decoded.reshape(28,28))

loss = tf.keras.losses.mse(decoded, img)
print(f"Loss: {np.mean(loss)}")
