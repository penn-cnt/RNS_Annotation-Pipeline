import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Conv1D, MaxPooling1D, BatchNormalization,GlobalAveragePooling2D, Input, Reshape, UpSampling2D
from keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.optimizers import schedules, SGD, Adam
import tensorflow as tf
from sklearn import preprocessing
from matplotlib import pyplot as plt
from keras import backend as K
from keras.applications import xception

def create_xception_model(input_shape=None):
    xception_model = xception.Xception(include_top=False,
        weights="imagenet",
        pooling='max',
        classes=128,
        classifier_activation=None,input_shape=input_shape)
    return xception_model

def NN(input_shape = None):

    channel_n = input_shape[1]
    length = input_shape[0]
    xception_model = create_xception_model(input_shape=(length, channel_n, 3))
    model = Sequential()
    model.add(Input(shape=(length,channel_n)))
    model.add(Reshape((length,channel_n,1)))
    model.add(Dense(3))
    model.add(Reshape((length,channel_n,3)))
    model.add(xception_model)
    model.add(Dense(128,activation='relu'))
    model.add(Dense(2,activation='softmax'))
    model.build()
    opt = Adam(learning_rate=1e-5, beta_1=0.995)
    callback = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",min_delta=1e-6,patience=10,restore_best_weights=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model

