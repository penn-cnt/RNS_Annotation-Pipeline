import tensorflow as tf
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model


class Autoencoder(Model):
    def __init__(self, length, channel):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(length, channel)),
            layers.Reshape((length, channel, 1)),
            layers.Conv2D(2, (1, 1), activation='relu', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(4, (1, 1), activation='relu', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(4, (1, 1), activation='relu', padding='same', strides=1),
            layers.BatchNormalization(),
            layers.Conv2D(4, (1, 1), activation='sigmoid', padding='same', strides=1),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(4),
            layers.Conv2D(4, (1, 1), activation='relu', padding='same', strides=1),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(2, (1, 1), activation='relu', padding='same', strides=1),
            layers.UpSampling2D((2, 2)),
            layers.Conv2D(1, (1, 1), activation='linear', padding='same', strides=1),
            layers.Reshape((length, channel))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def get_latent_embedding(encoder, data):
    latent_embedding = encoder.predict(data)
    latent_embedding = latent_embedding[:, :, 0, :]
    return latent_embedding
