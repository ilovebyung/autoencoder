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
for x, y in data:
    if y == 2:
        train.append(x)

train_images = np.array(train)
train_images = train_images.astype('float32') / 255.

# test
test = []
data = zip(x_test, y_test)
for x, y in data:
    if y == 2:
        test.append(x)

test_images = np.array(test)
test_images = test_images.astype('float32') / 255.


# Define a convolutional Autoencoder
latent_dim = 64 * 2


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu',
                          padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(
                8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(
                16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

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

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

# calculate loss and threshold
encoded_imgs = autoencoder.encoder(x_train).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
loss = tf.keras.losses.mse(decoded_imgs, x_train)
threshold = np.mean(loss) + np.std(loss)
print("Threshold: ", threshold)

# ## another option: images == 20
# history = autoencoder.fit(train_images, train_images,
#                 epochs=10,
#                 shuffle=True,
#                 validation_data=(test_images, test_images))

# encoded_imgs = autoencoder.encoder(test_images).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


# plot history
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()


# Display original and reconstruction
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
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
img = cv2.imread('test.jpg', 0)
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
