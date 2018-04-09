"""Shared inputs and interfaces for GAN model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import shared_utils

class CapsuleGANModel(object):
  """Main summarizer model class with learning and inference graphs."""

  def __init__(self, hps):
    self.hps=hps

 def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""

    tf.logging.info('Building graph...')

