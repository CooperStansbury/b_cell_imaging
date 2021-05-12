import pandas as pd
import numpy as np
import os
from math import sqrt
import numba
from scipy.spatial import distance_matrix


import skimage
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import reconstruction
from skimage import color, morphology
from skimage import exposure
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

DPI = 120


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
        'Ch2-T1' : {'gain' : 0.99, 'dilate' : True, 'tophat': True, 'equalize': -1},
        'ChS2-T2' : {'gain' : 1.2, 'dilate' : True, 'tophat': True, 'equalize': 0.2},
        'Ch1-T4' : {'gain' : 1.1, 'dilate' : True, 'tophat': True, 'equalize': -1},
        'ChS1-T3' : {'gain' : 0.8, 'dilate' : True, 'tophat': True, 'equalize': -1},
    }

    channel_params = params[channel]

    # and convert to normalize greyscale
    image = rgb2gray(image_in) # convert image to grey scale

    # histogram equalization
    if channel_params['equalize'] > 0:
        image = exposure.equalize_adapthist(image, clip_limit=channel_params['equalize'])

    # Logarithmic  pixel intensity correction
    image = exposure.adjust_log(image, channel_params['gain'])

    # dialate the image
    if channel_params['dilate']:
        seed = np.copy(image)
        seed[1:-1, 1:-1] = image.min()
        mask = image
        image = image - reconstruction(seed, mask, method='dilation')

    # top hatting small objects
    if channel_params['tophat']:
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
        'Ch2-T1' : {'min_sigma' : 7, 'max_sigma' : 20, 'num_sigma' : 3, 'threshold' : 0.09},
        'ChS2-T2' : {'min_sigma' : 7, 'max_sigma' : 20, 'num_sigma' : 5, 'threshold' : 0.1},
        'Ch1-T4' : {'min_sigma' : 7, 'max_sigma' : 20, 'num_sigma' : 3, 'threshold' : 0.12},
        'ChS1-T3' : {'min_sigma' : 7, 'max_sigma' : 20, 'num_sigma' : 3, 'threshold' : 0.15},
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


def plot_row(row, save=True):
    """A function to create three images based on an input 
    images. (1) the raw image, (2) the processed image, (3)
    the image overlaid with blobs.

    args:
        : row (row of pd.DataFrame)
        : save (bool): if true, saves an image for future ref
    """

    img = row['image']
    filname = row['filename']
    channel = row['channel']

    matplotlib.rcParams['figure.dpi'] = DPI
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(20, 20))
    ax1.imshow(img) 
    ax1.set_title(f"Raw Image: {filname}")

    img = preprocess_image(img, channel)
    ax2.imshow(img, cmap='magma')
    ax2.set_title("Processed Image")

    blobs = get_LoG_blobs(img, channel)
    ax3.imshow(img)

    for blob in blobs:
        y, x, r = blob
        c = plt.Circle((x, y), r, color='red', linewidth=2, fill=False)
        ax3.add_patch(c)

    ax3.set_title(f"N cells: {len(blobs)}")

    if save:
        outpath = f"figs/{filname}.png"
        plt.savefig(outpath, bbox_inches='tight') 
        print(f"Saved: {outpath}")


def plot_contour(row, save=True):
    """A function to plot the contour of a given channel
    
    args:
        :row (pd.Series row)
        : save (bool): if true, will save plot
    """
    image = rgb2gray(row['image'])
    filname = row['filename']

    fig, ax = plt.subplots(figsize=(5, 5))
    qcs = ax.contour(image, origin='image')
    ax.set_title(f'{filname} Contour plot')
    plt.show()

    if save:
        outpath = f"figs/{filname}_contour.png"
        plt.savefig(outpath, bbox_inches='tight') 
        print(f"Saved: {outpath}")


def plot_Multi_Otsu(row, classes=2, save=False):
    """A function to use binary Multi-Otsu thresholding
    """
    image = preprocess_image(row['image'], row['channel'])

    filname = row['filename']
    fig, ax = plt.subplots(figsize=(5, 5))

    thresholds = filters.threshold_multiotsu(image, classes=classes)
    regions = np.digitize(image, bins=thresholds)

    cells = image > thresholds[0]
    labeled_cells = measure.label(cells)

    ax.imshow(regions)
    ax.set_title(f'Multi-Otsu thresholding (n = {labeled_cells.max()})')
    ax.axis('off')
    plt.show()

    if save:
        outpath = f"figs/{filname}_multi_otsu.png"
        plt.savefig(outpath, bbox_inches='tight') 
        print(f"Saved: {outpath}")

        
        
def get_close_points(blobs1, blobs2, threshold):
    """A function to find points which are closer
    
    args:
        : blobs1 (array): blobs from group 1
        : blobs2 (array): blobs from group 2
        : threshold (float): threshold for points considered close
    
    returns:
        : close_points (np.array): indices of blob1 and blob2 considered close
    """
    
    # compute full euclidean distance matrix
    dists = distance_matrix(blobs1, blobs2)
    
    # # histogram of dists to choose the threshold 
    # sns.histplot(np.triu(dists, k=1).flatten(), bins=100)

    close_points = np.argwhere((dists < threshold) & (dists != 0))
    return close_points


def plot_overlap(df, color1, color2, threshold=10, save=False):
    """A function to resolve overlapping cells from each of 4 channels 
    
    args:
        : df (pd.DataFrame): a subset with detected blobs for each channel
        : color1 (str): color one
        : color2 (str):L color two
        : threshold (int): the distance threshold for points to be condisidered close
        : save (bool): is true, saves the image
    """
    c1 = np.asarray(df[df['channel_color'] == color1]['LoG_blobs'].iloc[0])
    c2 = np.asarray(df[df['channel_color'] == color2]['LoG_blobs'].iloc[0])
    
    group = df['group'].unique()
    day = df['day'].unique()
    
    close_pts = get_close_points(c1[:, 0:2], c2[:, 0:2], threshold)
    
    matplotlib.rcParams['figure.dpi'] = DPI
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim([0,1024])
    ax.set_ylim([0,1024])
    
    for i, j in close_pts:
        ax.scatter(c1[i, 1], c1[i, 0], 
           s=c1[i, 2]**2, 
           marker='o', 
           edgecolor='white',
           alpha=0.5,
           color=color1)
        
        ax.scatter(c2[j, 1], c2[j, 0], 
           s=c2[j, 2]**2, 
           marker='o', 
           edgecolor='white',
           alpha=0.5,
           color=color2)
        
    if save:
        outpath = f"figs/{group}_day_{day}_{c1}{c2}_overlap.png"
        plt.savefig(outpath, bbox_inches='tight') 
        print(f"Saved: {outpath}")