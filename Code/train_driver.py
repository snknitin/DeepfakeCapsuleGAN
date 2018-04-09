from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pprint import pprint

import os
import time
import math
import tensorflow as tf
import numpy as np
from tensorflow.python.client import timeline
from tensorflow.python.framework import test_util
import shared_utils as su
import data
import tf_lib
from datetime import datetime


def get_hps(base_dir, data_dir):
  hps= tf_lib.HParams(
      init_scale=0.08,
      learning_rate=1e-5,
      epsilon=1e-4,  # epsilon for Adam optimizer
      max_grad_norm=2.0,
      max_epoch=2,
      max_iters=4,
      max_max_epoch=13,
      hidden_size=512,
      batch_size=64,
      pass_lstm_states=True,
      use_separate_embeddings=False,
      use_separate_encoders=False,
      use_gpu=True,
      round_robin_gpu=True,
      max_gpus=4,
      colocate_gradients=False,
      base_dir=base_dir,
      data_dir=data_dir,
      log_device_placement=False,
      allow_mem_growth=False,

  )


  hps.train_mnist_files_path = os.path.join(data_dir, "train/train_mnist/")
  hps.train_cifar_files_path = os.path.join(data_dir, "train/train_cifar/")
  hps.train_celeba_files_path = os.path.join(data_dir, "train/train_celeba/")

  hps.dev_mnist_files_path = os.path.join(data_dir, "dev/dev_mnist/")
  hps.dev_cifar_files_path = os.path.join(data_dir, "dev/dev_cifar/")
  hps.dev_celeba_files_path = os.path.join(data_dir, "dev/dev_celeba/")

  hps.eval_mnist_files_path = os.path.join(data_dir, "eval/eval_mnist/")
  hps.eval_cifar_files_path = os.path.join(data_dir, "eval/eval_cifar/")
  hps.eval_celeba_files_path = os.path.join(data_dir, "eval/eval_celeba/")


  return hps


