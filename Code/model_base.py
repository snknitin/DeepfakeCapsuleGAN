"""Shared inputs and interfaces for GAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shared_utils as su
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate, Multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model

from keras.preprocessing.image import ImageDataGenerator
from keras import callbacks
import sys
import os
import glob
from tqdm import tqdm


class CapsuleGANModel(object):
    """Main summarizer model class with learning and inference graphs."""

    def __init__(self, hps,shape):
        self.hps=hps
        self.shape=shape
        self.D_L_REAL = []
        self.D_L_FAKE = []
        self.D_L = []
        self.D_ACC = []
        self.G_L = []

    def build_generator(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""

        tf.logging.info('Building generator...')
        """
            Generator follows the DCGAN architecture and creates generated image representations through learning.
            """

        noise_shape = (100,)
        x_noise = Input(shape=noise_shape)

        # we apply different kernel sizes in order to match the original image size

        if (self.shape[0] == 28 and self.shape[1] == 28):
            x = Dense(128 * 7 * 7, activation="relu")(x_noise)
            x = Reshape((7, 7, 128))(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D()(x)
            x = Conv2D(128, kernel_size=3, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D()(x)
            x = Conv2D(64, kernel_size=3, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Conv2D(1, kernel_size=3, padding="same")(x)
            gen_out = Activation("tanh")(x)

            return Model(x_noise, gen_out)

        if (self.shape[0] == 32 and self.shape[1] == 32):
            x = Dense(128 * 8 * 8, activation="relu")(x_noise)
            x = Reshape((8, 8, 128))(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D()(x)
            x = Conv2D(128, kernel_size=3, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = UpSampling2D()(x)
            x = Conv2D(64, kernel_size=3, padding="same")(x)
            x = Activation("relu")(x)
            x = BatchNormalization(momentum=0.8)(x)
            x = Conv2D(3, kernel_size=3, padding="same")(x)
            gen_out = Activation("tanh")(x)

            return Model(x_noise, gen_out)

    def build_discriminator(self):
        """Add the placeholders, model, global step, train_op and summaries to the graph"""

        tf.logging.info('Building discriminator...')
        # depending on dataset we define input shape for our network
        img = Input(shape=(self.shape[0], self.shape[1], self.shape[2]))

        # first typical convlayer outputs a 20x20x256 matrix
        x = Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', name='conv1')(img)
        x = LeakyReLU()(x)

        # original 'Dynamic Routing Between Capsules' paper does not include the batch norm layer after the first conv group
        x = BatchNormalization(momentum=0.8)(x)

        """
        NOTE: Capsule architecture starts from here.
        """
        #
        # primarycaps coming first
        #

        # filters 256 (n_vectors=8 * channels=32)
        x = Conv2D(filters=8 * 32, kernel_size=9, strides=2, padding='valid', name='primarycap_conv2')(x)

        # reshape into the 8D vector for all 32 feature maps combined
        # (primary capsule has collections of activations which denote orientation of the digit
        # while intensity of the vector which denotes the presence of the digit)
        x = Reshape(target_shape=[-1, 8], name='primarycap_reshape')(x)

        # the purpose is to output a number between 0 and 1 for each capsule where the length of the input decides the amount
        x = Lambda(su.squash, name='primarycap_squash')(x)
        x = BatchNormalization(momentum=0.8)(x)

        #
        # digitcaps are here
        #
        """
        NOTE: My approach is a simplified version of digitcaps i.e. without expanding dimensions into
        [None, 1, input_n_vectors, input_dim_capsule (feature maps)]
        and tiling it into [None, num_capsule, input_n_vectors, input_dim_capsule (feature maps)].
        Instead I replace it with ordinary Keras Dense layers as weight holders in the following lines.
    
        ANY CORRECTIONS ARE APPRECIATED IN THIS PART, PLEASE SUBMIT PULL REQUESTS!
        """
        x = Flatten()(x)
        # capsule (i) in a lower-level layer needs to decide how to send its output vector to higher-level capsules (j)
        # it makes this decision by changing scalar weight (c=coupling coefficient) that will multiply its output vector and then be treated as input to a higher-level capsule
        #
        # uhat = prediction vector, w = weight matrix but will act as a dense layer, u = output from a previous layer
        # uhat = u * w
        # neurons 160 (num_capsules=10 * num_vectors=16)
        uhat = Dense(160, kernel_initializer='he_normal', bias_initializer='zeros', name='uhat_digitcaps')(x)

        # c = coupling coefficient (softmax over the bias weights, log prior) | "the coupling coefficients between capsule (i) and all the capsules in the layer above sum to 1"
        # we treat the coupling coefficiant as a softmax over bias weights from the previous dense layer
        c = Activation('softmax', name='softmax_digitcaps1')(
            uhat)  # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one

        # s_j (output of the current capsule level) = uhat * c
        c = Dense(160)(c)  # compute s_j
        x = Multiply()([uhat, c])
        """
        NOTE: Squashing the capsule outputs creates severe blurry artifacts, thus we replace it with Leaky ReLu.
        """
        s_j = LeakyReLU()(x)

        #
        # we will repeat the routing part 2 more times (num_routing=3) to unfold the loop
        #
        c = Activation('softmax', name='softmax_digitcaps2')(
            s_j)  # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
        c = Dense(160)(c)  # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)

        c = Activation('softmax', name='softmax_digitcaps3')(
            s_j)  # softmax will make sure that each weight c_ij is a non-negative number and their sum equals to one
        c = Dense(160)(c)  # compute s_j
        x = Multiply()([uhat, c])
        s_j = LeakyReLU()(x)

        pred = Dense(1, activation='sigmoid')(s_j)

        return Model(img, pred)