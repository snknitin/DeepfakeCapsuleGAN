from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
from tqdm import tqdm
import glob
import scipy.misc
import math
import sys

import skimage
from skimage import data, color, exposure
from skimage.transform import resize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt





def test(hps):
    """
    Get the inception scores for the generated images from the gan
    :param hps:
    :return:
    """
    pass

