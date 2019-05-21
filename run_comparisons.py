#!/usr/bin/env python3

import os
import pickle
import sys
import time
sys.path.append('/home/dan/burkhardt/software/AAnet/')

import AAnet
import network
import AAtools
import tensorflow as tf

from sklearn.decomposition import PCA, NMF
from sklearn.manifold import MDS

import matplotlib.pyplot as plt
import matplotlib as mpl

font = {'size'   : 12}
mpl.rc('font', **font)

import numpy as np
import pandas as pd

import itertools

from skimage.transform import AffineTransform, warp

import scipy
import spams as sp

from PCHA import PCHA

import javadi


def calc_MSE(m1, m2):
    '''Finds 1-to-1 mapping of rows between m1 and m2 with lowest MSE and returns both the permutation.

    Parameters
    ----------
    m1 : [samples, features]
        Array 1
    m2 : [samples, features]
        Array 2

    Returns
    -------
    mse: float
        Mean squared error between the m1 and m2
    m1_idx: array
        Permutation index for m1
    m2_idx: array
        Permutation index for m2
    '''

    D = scipy.spatial.distance.cdist(m1, m2)
    m1_idx, m2_idx = scipy.optimize.linear_sum_assignment(D)
    mse = (np.square(m1[m1_idx,:] - m2[m2_idx,:])).mean()
    return mse, m1_idx, m2_idx

def run_AA(data, n_archetypes, true_archetypal_coords=None, true_archetypes=None, method='PCHA',
                n_subsample=None, n_batches=40000, latent_noise=0.05,
                arch=[1024,512,256,128], seed=42):
    """Runs Chen at al. 2014 on input data and calculates errors on the
    data in the archetypal space and the error between the learned vs true
    archetypes.

    Parameters
    ----------
    data : [samples, features]
        Data in the feature space
    true_archetypal_coords : [samples, archetypes]
        Ground truth archetypal coordinates. Rows must sum to 1.
    true_archetypes : [archetypes, features]
        Ground truth archetypes in the feature space
    n_archetypes : int
        Number of archetypes to be learned
    n_archetypes : int
        Number of observations to subsample for testing
    method : ['PCHA', 'kernelPCHA', 'Chen', 'Javadi', 'NMF', 'PCHA_on_AE', 'AAnet']
        The method to use for archetypal analysis
    n_subsample : int
        Number of data points to subsample
    seed : int
        Random seed
    batches : int
        Number of batches used to train AAnet or AutoEncoder
    Returns
    -------
    mse_archetypes: float
        Mean squared error between the learned archetypes and the ground
        truth archetypes as calculated in the feature space
    mse_encoding: float
        Mean squared error between the true coordinates of the data in the
        archetypal space and the coordinates in the learned space
    new_archetypal_coords: [samples, archetypes]
        Learned encoding of the samples in the archetypal space
    new_archetypes: [archetypes, features]
        Learned archetypes in the feature space
    """
    tic = time.time()
    # Select a subsample of the data
    np.random.seed(seed)
    if n_subsample is not None:
        r_idx = np.random.choice(data.shape[0], n_subsample, replace=False)
        data = data[r_idx,:] # otherwise really slow
        true_archetypal_coords = true_archetypal_coords[r_idx,:]

    if method == 'Chen':
        '''AA as implemented in Chen et al. 2014 https://arxiv.org/abs/1405.6472'''
        new_archetypes, new_archetypal_coords, _ = sp.archetypalAnalysis(
                    np.asfortranarray(data.T), p=n_archetypes, returnAB=True, numThreads=-1)

        # Fix transposition
        new_archetypal_coords = new_archetypal_coords.toarray().T
        new_archetypes = new_archetypes.T
    elif method == 'Javadi':
        '''AA as implemented in Javadi et al. 2017 https://arxiv.org/abs/1705.02994'''

        new_archetypal_coords, new_archetypes, _, _ = javadi.acc_palm_nmf(data,
                                        r=n_archetypes, maxiter=25, plotloss=False,
                                        ploterror=False)
    elif method == 'PCHA':
        '''Principal convex hull analysis as implemented by Morup and Hansen 2012.
        https://www.sciencedirect.com/science/article/pii/S0925231211006060 '''
        new_archetypes, new_archetypal_coords, _, _, _ = PCHA(data.T, noc=n_archetypes)
        new_archetypes = np.array(new_archetypes.T)
        new_archetypal_coords = np.array(new_archetypal_coords.T)

    elif method == 'kernelPCHA':
        '''PCHA in a kernel space as described by Morup and Hansen 2012.
        https://www.sciencedirect.com/science/article/pii/S0925231211006060 '''
        D = scipy.spatial.distance.pdist(data)
        D = scipy.spatial.distance.squareform(D)
        sigma = np.std(D)
        #K = np.exp(-((D**2)/sigma))
        K = data @ data.T
        _, new_archetypal_coords, C, _, _ = PCHA(K, noc=n_archetypes)
        new_archetypes = np.array(data.T @ C).T
        new_archetypal_coords = np.array(new_archetypal_coords.T)

    elif method == 'NMF':
        '''Factor analysis using non-negative matrix factorization (NMF)'''
        nnmf = NMF(n_components=n_archetypes, init='nndsvda', tol=1e-4, max_iter=1000)
        new_archetypal_coords = nnmf.fit_transform(data - np.min(data))
        new_archetypes = nnmf.components_

    elif method == 'PCHA_on_AE':
        ##############
        # MODEL PARAMS
        ##############
        noise_z_std = 0
        z_dim = arch
        act_out = tf.nn.tanh
        input_dim = data.shape[1]

        enc_AE = network.Encoder(num_at=n_archetypes, z_dim=z_dim)
        dec_AE = network.Decoder(x_dim=input_dim, noise_z_std=noise_z_std, z_dim=z_dim, act_out=act_out)

        # By setting both gammas to zero, we arrive at the standard autoencoder
        AE = AAnet.AAnet(enc_AE, dec_AE, gamma_convex=0, gamma_nn=0)
        ##########
        # TRAINING
        ##########
        # AE
        AE.train(data, batch_size=256, num_batches=n_batches)
        latent_encoding = AE.data2z(data)

        # PCHA learns an encoding into a simplex
        new_archetypes, new_archetypal_coords, _, _, _ = PCHA(latent_encoding.T, noc=n_archetypes)
        new_archetypes = np.array(new_archetypes.T)
        new_archetypal_coords = np.array(new_archetypal_coords.T)

        # Decode ATs
        new_archetypes = AE.z2data(new_archetypes)

    elif method == 'AAnet':
        ##############
        # MODEL PARAMS
        ##############

        noise_z_std = latent_noise
        z_dim = arch
        act_out = tf.nn.tanh
        input_dim = data.shape[1]

        enc_net = network.Encoder(num_at=n_archetypes, z_dim=z_dim)
        dec_net = network.Decoder(x_dim=input_dim, noise_z_std=noise_z_std, z_dim=z_dim, act_out=act_out)
        model = AAnet.AAnet(enc_net, dec_net)

        ##########
        # TRAINING
        ##########

        model.train(data, batch_size=256, num_batches=n_batches)

        ###################
        # GETTING OUTPUT
        ###################

        new_archetypal_coords = model.data2at(data)
        new_archetypes = model.get_ats_x()
    else:
        raise ValueError('{} is not a valid method'.format(method))
    toc = time.time() - tic
    # Calculate MSE if given ground truth
    if true_archetypes is not None:
        mse_archetypes ,_ ,_ = calc_MSE(new_archetypes, true_archetypes)
    else:
        mse_archetypes = None
    if true_archetypal_coords is not None:
        mse_encoding ,_ ,_ = calc_MSE(new_archetypal_coords.T, true_archetypal_coords.T)
    else:
        mse_encoding = None

    return mse_archetypes, mse_encoding, new_archetypal_coords, new_archetypes, toc


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

    # Finally, apply rotation
    shifted = transform.rotate(shifted, rotation)

    return shifted, weights
    #plt.matshow(blank_canvas)
