import os
from time import time
import pandas as pd
import numpy as np
from keras.datasets import mnist, cifar10









def load_dataset(hps):
    if hps.module == 'mnist':
        width, height, channels=hps.train_mnist_dimensions

        # load MNIST data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

    if hps.module == 'cifar10':
        # load CIFAR10 data
        width, height, channels = hps.train_cifar_dimensions
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()

        # rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

    if hps.module == 'celeba':
        # load CelebA data
        width, height, channels = hps.train_celeba_dimensions
        (X_train, y_train), (X_test, y_test) = celeba.load_data()
        # rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5


    # defining input dims
    img_rows = width
    img_cols = height
    channels = channels
    img_shape = [img_rows, img_cols, channels]

    return X_train, img_shape
