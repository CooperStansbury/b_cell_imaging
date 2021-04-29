import pandas as pd
import numpy as np
import os
from math import sqrt
import numba

import skimage
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import reconstruction
from skimage import color, morphology

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


@numba.jit
def preprocess_image(image):
    """A function to perform a few preprocessing steps before 
    running blob detection"""
    # and convert to normalize greyscale
    image_gray = rgb2gray(image) # convert image to grey scale

    # # use hard threshold
    # u, s, vt = np.linalg.svd(image_gray)
    # r = (np.sqrt(4) / 3) * np.median(s)
    # s_ind = np.argwhere(s >= r)
    # k = np.max(s_ind)
    # image = np.dot(u[:,0:k] * s[0:k,], vt[0:k,])

    # top hatting small objects
    selem =  morphology.disk(1)
    res = morphology.white_tophat(image, selem)
    image = image - res

    # dialate the image
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    image = image - reconstruction(seed, mask, method='dilation')

    return image
    

def get_LoG_blobs(image, params):
    """A function run blob detection on an image.
    
    args:
        : image (np.array): an input image, assumed 2D
        : params (dict): parameters for https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    returns:
        : blobs (np.array): first two columns are center coords, 3rd is radius 
    """
    blobs_log = blob_log(image, **params)

    # compute the radii
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    return blobs_log


def plot_LoG_blobs(image, blobs):
    """A function to plot blobs and an image in the same window """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(image, cmap='icefire')

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=1, fill=False)
        ax.add_patch(c)

    return plt