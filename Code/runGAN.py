"""This is the main file to run the Capsule GAN model"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import sys
import time
import os
import tensorflow as tf
import numpy as np
import random
import train_driver as td


FLAGS = tf.app.flags.FLAGS

# Where to find data and where to write outputs

tf.app.flags.DEFINE_string('base_dir', '', """Base directory to work out of and write output.""")
tf.app.flags.DEFINE_string('data_dir', '', """Base directory to from which to read data.""")

# Important settings
tf.app.flags.DEFINE_string('module', 'mnist', 'must be one of mnist,cifar/celeba')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of train/val/test')
# Where to save output
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')


tf.app.flags.DEFINE_integer('random_seed', 1337, """Random seed for all RNGs (python, tensorflow, numpy).""")
tf.app.flags.DEFINE_integer('num_epochs', '100', 'Number of epochs in the training')





def main(unused_argv):

    if len(unused_argv) != 1:  # prints a message if you've entered flags incorrectly
        raise Exception("Problem with flags: %s" % unused_argv)

    if FLAGS.base_dir == "" or not os.path.isdir(FLAGS.base_dir):
        raise ValueError("Must provide a valid base directory for driver.")

    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
    tf.logging.info('Starting capsule gan in %s mode...', (FLAGS.mode))

    mode_modules = {
        "mnist": td.mnist,
        "cifar": td.cifar,
        "celeba":td.celeba}
    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)

    module_dir = os.path.join(FLAGS.log_root, FLAGS.module)  # make a subdir of the root dir for chosen module data

    # Create directory for module only after running in train mode
    if not os.path.exists(module_dir):
        if FLAGS.mode=="train":
          os.makedirs(module_dir)
        else:
          raise Exception("Module directory doesn't exist in %s . Run in train mode to create it." % (FLAGS.log_root))

    hps = td.get_hps(FLAGS.base_dir, FLAGS.data_dir)
    hps.mode= FLAGS.mode
    hps.module_dir=module_dir


    tf.logging.info('Reading from vocabulary file...')

    # Call the model
    if FLAGS.module in mode_modules:
        print("Module: ",FLAGS.module)
        print("Mode: ",FLAGS.mode)
        module = mode_modules[FLAGS.module]
        module(hps)

    else:
        raise ValueError("Unknown mode: %s" % FLAGS.mode)


if __name__ == '__main__':
    tf.app.run()