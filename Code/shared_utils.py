import os
import pandas as pd
import numpy as np
import h5py
from keras import backend as K
# import skimage
# from skimage import data, color, exposure
# from skimage.transform import resize
import matplotlib.pyplot as plt




def save_fig(save_path,fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(save_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.axis('off')
    plt.savefig(path,transparent = True, bbox_inches = 'tight', pad_inches =-0.1, format=fig_extension, dpi=resolution)

def save_imgs(dataset_title,generator, epoch,hps):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)

    # rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0

    # iterate in order to create a subplot
    for i in range(r):
        for j in range(c):
            if dataset_title == 'mnist':
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
            elif dataset_title == 'cifar10':
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
            elif dataset_title == 'celeba':
                axs[i, j].imshow(gen_imgs[cnt, :, :, :])
                axs[i, j].axis('off')
                cnt += 1
            else:
                print('Please indicate the image options.')
    path = os.path.join(hps.img_dir, 'images_{}_{}'.format(dataset_title,epoch))
    # if not os.path.exists(path):
    #     os.makedirs(path)

    fig.savefig(path)
    plt.close()

# squash function of capsule layers, borrowed from Xifeng Guo's implementation of Keras CapsNet `https://github.com/XifengGuo/CapsNet-Keras`
def squash(vectors, axis=-1):
    """
    The non-linear activation used in Capsule. It drives the length of a large vector to near 1 and small vector to 0
    :param vectors: some vectors to be squashed, N-dim tensor
    :param axis: the axis to squash
    :return: a Tensor with same shape as input vectors
    """
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors


def load_data():
    """
    Loads the CelebA dataset from the hdf5 file and processes it
    :return:
    """

    with h5py.File(os.path.join(os.path.dirname(os.getcwd()),"Data/celebA/CelebA_32_data.h5"), "r") as hf:
        # Loading the data as floats
        X_real_train = hf["data"][:].astype(np.float32)
        # Transpose to make channels the last dimension
        X_real_train = X_real_train.transpose(0, 2, 3, 1)
        # Normalizing the pixels
        X_real_train = (X_real_train- 127.5) / 127.5
        np.random.shuffle(X_real_train)
        # Split to 80%
        return X_real_train[:162080]


def save_gen_img(dataset_title,generator,num_images):
    IMAGES_PATH = os.path.join(os.path.join(os.path.dirname(os.getcwd()),"Data/"), "static/{}".format(dataset_title))
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)
    cnt=0
    while cnt<num_images:

        if dataset_title == 'mnist':
            # rescale images 0 - 1
            noise = np.random.normal(0, 1, (32,100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5
            plt.imshow(gen_imgs[1, :, :, 0], cmap='gray')
            save_fig(IMAGES_PATH,"mnist_gen_{}".format(cnt))
            cnt += 1
        elif dataset_title == 'cifar10':
            # rescale images 0 - 1
            noise = np.random.normal(0, 1, (32,100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5
            plt.imshow(gen_imgs[1, :, :, :])
            save_fig(IMAGES_PATH,"cifar10_gen_{}".format(cnt))
            cnt += 1
        elif dataset_title == 'celeba':
            # rescale images 0 - 1
            noise = np.random.normal(0, 1, (32,100))
            gen_imgs = generator.predict(noise)
            gen_imgs = 0.5 * gen_imgs + 0.5
            plt.imshow(gen_imgs[1, :, :, :])
            save_fig(IMAGES_PATH,"celeba_gen_{}".format(cnt))
            cnt += 1
        else:
            print('Please indicate the image options.')
