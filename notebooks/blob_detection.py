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
from skimage import exposure

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns


def preprocess_image(image_in, channel):
    """A function to perform a few preprocessing steps before 
    running blob detection
    
    args:
        : image_in (nd.array): raw image for processing
        : channel (str): for channel specific parameters

    return:
        : img (nd.array): a processed image
    """

    params = {
        'Ch2-T1' : {'gain' : 0.99},
        'ChS2-T2' : {'gain' : 2.7},
        'Ch1-T4' : {'gain' : 1.1},
        'ChS1-T3' : {'gain' : 0.8},
    }

    channel_params = params[channel]

    # and convert to normalize greyscale
    image = rgb2gray(image_in) # convert image to grey scale

    # Logarithmic  pixel intensity correction
    image = exposure.adjust_log(image, channel_params['gain'])

    # dialate the image
    seed = np.copy(image)
    seed[1:-1, 1:-1] = image.min()
    mask = image
    image = image - reconstruction(seed, mask, method='dilation')

    # # use hard threshold
    # u, s, vt = np.linalg.svd(image_gray)
    # r = (np.sqrt(4) / 3) * np.median(s)
    # s_ind = np.argwhere(s >= r)
    # k = np.max(s_ind)
    # image = np.dot(u[:,0:k] * s[0:k,], vt[0:k,])

    # top hatting small objects
    selem = morphology.disk(1)
    res = morphology.white_tophat(image, selem)
    image = image - res


    return image
    

def get_LoG_blobs(image, channel):
    """A function run blob detection on an image.
    
    args:
        : image (np.array): an input image, assumed 2D
        : channel (str): parameters for https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.blob_log
    
    returns:
        : blobs (np.array): first two columns are center coords, 3rd is radius 
    """
    params = {
        'Ch2-T1' : {'min_sigma' : 5, 'max_sigma' : 20, 'num_sigma' : 3, 'threshold' : 0.1},
        'ChS2-T2' : {'min_sigma' : 5, 'max_sigma' : 20, 'num_sigma' : 5, 'threshold' : 0.09},
        'Ch1-T4' : {'min_sigma' : 5, 'max_sigma' : 20, 'num_sigma' : 3, 'threshold' : 0.12},
        'ChS1-T3' : {'min_sigma' : 8, 'max_sigma' : 20, 'num_sigma' : 3, 'threshold' : 0.17},
    }
    channel_params = params[channel]
    blobs_log = blob_log(image, **channel_params)

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