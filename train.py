""" Train your model """
import argparse
from PIL.Image import NEAREST # native imports 

import numpy as np # third party imports
import rasterio as rs
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dropout, Activation
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, UpSampling2D 
from tensorflow.keras.metrics import RootMeanSquaredError

from utils.helper import * # local imports 


def demoPlot(month):
    path = "heatmap_data/wsi_1981-1990_0" + str(month) + ".tif"
    demo = np.array(rs.open(path).read(1))
    for i in range(demo.shape[0]):
        for j in range(demo.shape[1]):
            if demo[i, j] < -1000:
                demo[i, j] = 0
    plt.imshow(demo, cmap="hot", interpolation="nearest")
    plt.show()

def preprocess_data():
    # Initializing empty arrays for the 24 heatmaps
    zeros = np.zeros((280, 720, 12))
    zeros2 = np.zeros((280, 720, 12))

    # Importing 1980-1990 heatmap data
    string = "heatmap_data/wsi_1981-1990_0"
    for z in range(9):
        test = string + str(z + 1) + ".tif"
        temp = rs.open(test)
        band1 = temp.read(1)
        arr = np.array(band1)
        zeros[:, :, z] = arr

    zeros[:, :, 9] = np.array(rs.open("heatmap_data/wsi_1981-1990_10.tif").read(1))
    zeros[:, :, 10] = np.array(rs.open("heatmap_data/wsi_1981-1990_11.tif").read(1))
    zeros[:, :, 11] = np.array(rs.open("heatmap_data/wsi_1981-1990_12.tif").read(1))

    zeros = np.where(zeros == np.min(zeros), 0, zeros)

    # Importing 2001-2010 heatmap data
    string = "heatmap_data/wsi_2001-2010_0"
    for z in range(9):
        test = string + str(z + 1) + ".tif"
        temp = rs.open(test)
        band1 = temp.read(1)
        arr = np.array(band1)
        zeros2[:, :, z] = arr

    zeros2[:, :, 9] = np.array(rs.open("heatmap_data/wsi_2001-2010_10.tif").read(1))
    zeros2[:, :, 10] = np.array(rs.open("heatmap_data/wsi_2001-2010_11.tif").read(1))
    zeros2[:, :, 11] = np.array(rs.open("heatmap_data/wsi_2001-2010_12.tif").read(1))

    zeros2 = np.where(zeros2 == np.min(zeros2), 0, zeros2)

    # Intializing new arrays to match formatting for Tensorflow
    NewZeros = np.zeros((18, 280, 720, 3))
    Y = np.zeros((18, 280, 720))

    projection = np.zeros((1, 280, 720, 3))
    for i in range(3):
        projection[0, :, :, i] = zeros2[:, :, 9 + i]

    for z in range(9):
        Y[z, :, :] = zeros[:, :, z + 3]

    for z in range(9):
        Y[z + 9, :, :] = zeros2[:, :, z + 3]

    for z in range(9):
        for y in range(3):
            NewZeros[z, :, :, y] = zeros[:, :, z + y]

    for z in range(9):
        for y in range(3):
            NewZeros[z + 9, :, :, y] = zeros2[:, :, z + y]

    # Splitting into training and testing data
    training_split = 0.833
    training_x = NewZeros[:15, :, :, :]
    training_y = Y[:15, :, :]
    testing_x = NewZeros[15:, :, :, :]
    testing_y = Y[15:, :, :]

    return training_x, training_y, testing_x, testing_y

def train_model(input_data, output_data):
    """
    This function trains a CNN to predict heatmaps of future water scarcity
    Inputs: 
        input_data (N, 280, 720, 3)
        output_data (N, 280, 720)
    """
    model = tf.keras.Sequential()
    model.add(Convolution2D(32,
                            3,
                            input_shape=input_data.shape[1:],
                            padding="same",
                            kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(32, 3, padding="same", kernel_initializer="he_normal"))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(32, 3, padding="same", kernel_initializer="he_normal"))
    model.add(Activation('relu'))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 3, padding="same", kernel_initializer="he_normal"))
    model.add(Activation('relu'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(input_data, output_data, batch_size=2, epochs=10,
            verbose=1, steps_per_epoch=8)

    return model

def test_model(model, testing_x, testing_y):
    return model.evaluate(testing_x, testing_y)

if __name__ == '__main__':

    training_x, training_y, testing_x, testing_y = preprocess_data()
    model = train_model(training_x, training_y)