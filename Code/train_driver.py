from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import os
import time
import math
import tensorflow as tf
import numpy as np
from keras import Model, Input
from keras.optimizers import Adam
from tensorflow.python.client import timeline
from tensorflow.python.framework import test_util
import shared_utils as su
import data
import tf_lib
import model_base as mb
from datetime import datetime
from sklearn.externals import joblib
import ujson
import pickle
import matplotlib.pyplot as plt

def get_hps(base_dir, data_dir):
  hps= tf_lib.HParams(
      init_scale=0.08,
      learning_rate=2e-4,
      epsilon=1e-4,  # epsilon for Adam optimizer
      beta_1=0.9,
      beta_2=0.999,
      batch_size=64,
      use_gpu=True,
      round_robin_gpu=True,
      max_gpus=4,
      colocate_gradients=False,
      base_dir=base_dir,
      data_dir=data_dir,
      log_device_placement=False,
      allow_mem_growth=False,

  )

  hps.train_mnist_dimensions = (28,28,1)
  hps.train_cifar_dimensions = (32,32,3)
  hps.train_celeba_dimensions = (32,32,3)

  return hps


def train(hps, epochs, save_interval=1000):
    half_batch = int(hps.batch_size / 2)
    dataset, shape = data.load_dataset(hps)
    # loss values for further plotting

    model=mb.CapsuleGANModel(hps,shape)
    discriminator = model.build_discriminator()
    generator = model.build_generator()
    discriminator.compile(loss='binary_crossentropy', optimizer=Adam(hps.learning_rate,hps.beta_1,hps.beta_2,hps.epsilon), metrics=['accuracy'])
    generator.compile(loss='binary_crossentropy', optimizer=Adam(hps.learning_rate,hps.beta_1,hps.beta_2,hps.epsilon))

    z = Input(shape=(100,))
    img = generator(z)
    discriminator.trainable = False
    valid = discriminator(img)
    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=Adam(hps.learning_rate,hps.beta_1,hps.beta_2,hps.epsilon))
    for epoch in range(epochs):

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # select a random half batch of images
        idx = np.random.randint(0, dataset.shape[0], half_batch)
        imgs = dataset[idx]

        noise = np.random.normal(0, 1, (half_batch, 100))

        # generate a half batch of new images
        gen_imgs = generator.predict(noise)

        # train the discriminator by feeding both real and fake (generated) images one by one
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)) * 0.9)  # 0.9 for label smoothing
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  Train Generator
        # ---------------------

        noise = np.random.normal(0, 1, (hps.batch_size, 100))

        # the generator wants the discriminator to label the generated samples
        # as valid (ones)
        valid_y = np.array([1] * hps.batch_size)

        # train the generator
        g_loss = combined.train_on_batch(noise, np.ones((hps.batch_size, 1)))

        # Plot the progress
        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))
        model.D_L_REAL.append(d_loss_real)
        model.D_L_FAKE.append(d_loss_fake)
        model.D_L.append(d_loss)
        model.D_ACC.append(d_loss[1])
        model.G_L.append(g_loss)

        # if at save interval => save generated image samples
        if epoch % (5*save_interval) == 0:
            su.save_imgs(hps.module,generator, epoch,hps)
        if epoch % (10*save_interval) == 0:
            generator.save(hps.module+'_gen_model_{}.h5'.format(epoch))
            discriminator.save(hps.module+'_dis_model_{}.h5'.format(epoch))
        # if epoch % (15*save_interval) == 0:
        #     # joblib.dump(model, "model_{}.pkl".format(epoch))
        #     with open("model_{}.json".format(epoch), 'w') as f:
        #         ujson.dump(model, f)
        #     f.close()
    plt.plot(model.D_L)
    plt.title('Discriminator results')
    plt.xlabel('Epochs')
    plt.ylabel('Discriminator Loss (blue), Discriminator Accuracy (orange)')
    plt.legend(['Discriminator Loss', 'Discriminator Accuracy'])
    su.save_fig("{}_DL".format(hps.module))

    plt.plot(model.G_L)
    plt.title('Generator results')
    plt.xlabel('Epochs')
    plt.ylabel('Generator Loss (blue)')
    plt.legend('Generator Loss')
    su.save_fig("{}_GL".format(hps.module))