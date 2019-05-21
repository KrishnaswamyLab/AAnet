#!/usr/bin/env python3

import os, pickle, time

import sys
sys.path.append('/home/dan/burkhardt/software/AAnet/')

import AAnet
import network
import AAtools
import tensorflow as tf

import run_comparisons

from sklearn.decomposition import PCA
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'size'   : 12}
mpl.rc('font', **font)

import numpy as np
import pandas as pd

import itertools
import scprep

from skimage.transform import AffineTransform, warp, rescale
from skimage import transform
import run_comparisons

def get_new_digit(curr_img, weights=None):
    if weights is None:
        # Get weights (aka the archetypal space)
        u = np.random.uniform(0,1, 4)
        e = -np.log(u)
        weights = e / np.sum(e)
    elif weights.sum() != 1:
        raise ValueError('`weights` must sum to 1.')

    # Transformation from the archetypal space to the latent space
    h_scale = 2.25 - (weights[2] * 1.5)
    v_scale = (weights[2] * 1.5) + .75
    rotation = 0#weights[2] * 45
    h_offset = (weights[1] * 20) - 10
    v_offset = (weights[0] * 20) - 10

    ## Apply rescaling and place rescaled image in a larger canvas
    img_tform = transform.rescale(curr_img, (v_scale, h_scale))# h_scale))

    #plt.matshow(img_tform)
    blank_canvas = np.zeros((84,84))

    # Handle horizontal rescaling
    h_start = (blank_canvas.shape[0] - img_tform.shape[0]) / 2
    h_delta = h_start % 1 # need to translate by this amount
    h_start = int(h_start // 1)
    h_stop  =  h_start + img_tform.shape[0]

    # Handle vertical rescaling
    v_start = (blank_canvas.shape[1] - img_tform.shape[1]) / 2
    v_delta = v_start % 1 # need to translate by this amount
    v_start = int(v_start // 1)
    v_stop  =  v_start + img_tform.shape[1]

    # Place Image on canvas
    blank_canvas[h_start:h_stop, v_start:v_stop] = img_tform

    # Translate image at subpixel resolution to make sure it's properly centered
    tform = AffineTransform(translation=(-h_delta - h_offset , -v_delta - v_offset))
    shifted = warp(blank_canvas, tform, mode='wrap', preserve_range=True)

    # Apply rotation
    shifted = transform.rotate(shifted, rotation)

    # Rescale
    shifted = rescale(shifted, 0.67)
    return shifted, weights
    #plt.matshow(blank_canvas)

dataset_zip = np.load('/home/dan/burkhardt/archetypal_analysis/files/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz', allow_pickle=True, encoding='latin1')

imgs = dataset_zip['imgs']
latents_values = dataset_zip['latents_values']
latents_classes = dataset_zip['latents_classes']
metadata = dataset_zip['metadata'][()]

# Define number of values per latents and functions to convert to indices
latents_sizes = metadata['latents_sizes']
latents_bases = np.concatenate((latents_sizes[::-1].cumprod()[::-1][1:],
                                np.array([1,])))

def latent_to_index(latents):
      return np.dot(latents, latents_bases).astype(int)

basic_sprites = []

for i in [0,1,2]:
    curr_latents = [0, i, 3, 0, 0, 0]

    indices_sampled = latent_to_index(curr_latents)
    imgs_sampled = imgs[indices_sampled]
    curr_sprite = imgs_sampled[2:30, 2:30]
    curr_sprite = curr_sprite / np.max(curr_sprite)
    basic_sprites.append(curr_sprite)

results = {}

for curr_sp in [0,1,2]:
    for seed in [7, 42, 111]:#, 333, 666]:
        np.random.seed(seed) # For running triplicates
        data = []
        true_archetypal_coords = []
        for i in range(15000):
            new_image, archetypal_coords = get_new_digit(basic_sprites[curr_sp])
            new_image = new_image.flatten()

            data.append(new_image)
            true_archetypal_coords.append(archetypal_coords)

        data = np.vstack(data)
        data = data / data.max()
        data = (data * 2) - 1

        true_archetypal_coords = np.vstack(true_archetypal_coords)

        true_archetypes = []
        for weights in np.eye(4):
            true_archetypes.append(get_new_digit(basic_sprites[curr_sp], weights)[0].flatten())
        true_archetypes = np.vstack(true_archetypes)

        print('Running comparisons on sprite {} with seed {}...'.format(curr_sp, seed))
        for method in ['PCHA_on_AE', 'AAnet']:
            print('    Starting {}...'.format(method))
            tic = time.time()
            curr_results = run_comparisons.run_AA(data,
                           true_archetypal_coords=true_archetypal_coords,
                           true_archetypes=true_archetypes,
                           n_archetypes=4,
                           method=method,
                           n_subsample=None,
                           arch=[512,256,128],
                           n_batches=40000)
            results[(method, curr_sp, seed)] = curr_results
            print('        Finished in {:.2f} seconds.'.format(curr_results[-1]))
        print('    Done!')
with open('/home/dan/burkhardt/archetypal_analysis/files/sprites_comparisons.15k.smAA.pkl', 'wb+') as f:
    pickle.dump(results, f)
