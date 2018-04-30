import os
import numpy as np
import h5py

from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt
import shared_utils as su






if __name__=="__main__":
    dataset_title="cifar10"
    #mnist_loc= "E:\\GIT_ROOT\\DeepfakeCapsuleGAN\\Data\\log\\mnist_train_02\\mnist\\saved_models\\mnist_gen_model_46000.h5"
    #cifar_loc= "E:\\GIT_ROOT\\DeepfakeCapsuleGAN\\Data\\log\\cifar_train_01\\cifar10\\saved_models\\cifar10_gen_model_132000.h5"
    cifar_loc="/mnt/nfs/scratch1/nsamala/capsuleGAN/DeepfakeCapsuleGAN/Data/log/cifar_train_01/cifar10/saved_models/cifar10_gen_model_132000.h5"
    #celeba_loc = "E:\\GIT_ROOT\\DeepfakeCapsuleGAN\\Data\\log\\celeba_train_01\\celeba\\saved_models\\celeba_gen_model_128000.h5"

    generator=load_model(cifar_loc)
    su.save_gen_img(dataset_title,generator,100)