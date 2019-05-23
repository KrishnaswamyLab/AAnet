#!/usr/bin/env python3

import os, pickle, time, itertools
from urllib.request import urlopen

import tensorflow as tf
import numpy as np

from sklearn.manifold import MDS
from skimage.transform import AffineTransform, warp
from skimage import transform

# AAnet imports
import AAnet
import network
import AAtools
import run_comparisons

def get_new_digit(curr_img, weights=None):
    '''Takes a sprite extracted from dSprites and performs a random of transformation dictacted by `weights`'''

    # Take a sample from a four dimensional simplex
    if weights is None:
        # Get weights (aka the archetypal space)
        u = np.random.uniform(0,1, 4)
        e = -np.log(u)
        weights = e / np.sum(e)
    elif weights.sum() != 1:
        raise ValueError('`weights` must sum to 1.')

    # Transformation from the archetypal space to the latent space
    v_offset = (weights[0] * 20) - 10
    h_offset = (weights[1] * 20) - 10
    h_scale = 2.25 - (weights[2] * 1.5)
    v_scale = (weights[2] * 1.5) + .75

    ## Apply aspect ratio rescaling
    img_tform = transform.rescale(curr_img, (v_scale, h_scale), multichannel=False)# h_scale))

    # Bigger canvas, image will be center prior to applying horizontal and vertical translations
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

    # Rescale image to 56x56 pixels
    shifted = transform.rescale(shifted, 0.67, multichannel=False)
    return shifted, weights
    #plt.matshow(blank_canvas)

# Download dSprites dataset
download_path = os.path.expanduser("~")
print('Saving data to: {}'.format(download_path))
data_url = 'https://github.com/deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
data_npz = os.path.join(download_path, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

if not os.path.isdir(os.path.join(download_path, "dSprites")):
    if not os.path.isdir(download_path):
        os.mkdir(download_path)
    if not os.path.isfile(data_npz):
        with urlopen(data_url) as url:
            print("Downloading data file...")
            # Open our local file for writing
            with open(data_npz, "wb") as handle:
                handle.write(url.read())

# Load dSprites (code from dSprites tutorial)
dataset_zip = np.load(data_npz, allow_pickle=True, encoding='latin1')

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

# Get four sprites to manually manipulate for tests
basic_sprites = []
for i in [0,1,2]:
    # Define latent features
    curr_latents = [0, i, 3, 0, 0, 0]
    # Get the image
    indices_sampled = latent_to_index(curr_latents)
    imgs_sampled = imgs[indices_sampled]
    curr_sprite = imgs_sampled[2:30, 2:30]
    curr_sprite = curr_sprite / np.max(curr_sprite) # Rescale to [0,1]
    basic_sprites.append(curr_sprite)

# Collect results
results = {}

for curr_sp in [0,1,2]:
    for seed in [0,1,2,3,4]:
        np.random.seed(seed) # For running triplicates
        data = []
        true_archetypal_coords = []
        # Sample 15000 images
        for i in range(15000):
            new_image, archetypal_coords = get_new_digit(basic_sprites[curr_sp])
            new_image = new_image.flatten()
            data.append(new_image)
            true_archetypal_coords.append(archetypal_coords)

        # Normalize images to [-1,1]
        data = np.vstack(data)
        data = data / data.max()
        data = (data * 2) - 1

        true_archetypal_coords = np.vstack(true_archetypal_coords)

        # Collect true archetypes of each digit manually
        true_archetypes = []
        for weights in np.eye(4):
            true_archetypes.append(get_new_digit(basic_sprites[curr_sp], weights)[0].flatten())
        true_archetypes = np.vstack(true_archetypes)

        # Run the comparisons
        print('Running comparisons on sprite {} with seed {}...'.format(curr_sp, seed))
        for method in ['PCHA', 'kernelPCHA',  'NMF', 'PCHA_on_AE', 'AAnet', 'Chen', 'Javadi']:
            print('    Starting {}...'.format(method))
            tic = time.time()
            if method == 'AAnet':
                arch = [1024,512,256,128]
            else:
                # Need to do this because for some reason the AE
                # won't train properly with 4 layers
                arch = [512,256,128]
            curr_results = run_comparisons.run_AA(data,
                           n_archetypes=4,
                           true_archetypal_coords=true_archetypal_coords,
                           true_archetypes=true_archetypes,
                           method=method,
                           n_subsample=None,
                           arch=arch,
                           n_batches=40000)
            results[(method, curr_sp, seed)] = curr_results
            print('        Finished in {:.2f} seconds.'.format(curr_results[-1]))
        print('    Done!')

with open(os.path.join(download_path, 'comparison_results.pkl'), 'wb+') as f:
    pickle.dump(results, f)

'''
Results can be loaded with the following code:

import pandas as pd
import seaborn as sns

with open(os.path.join(download_path, 'comparison_results.pkl'), 'rb+') as f:
    results = pickle.load(f)

MSE = []
for curr_sp, seed, method in results:
    mse_archetypes, mse_encoding, new_archetypal_coords, new_archetypes,\
    time = results[(method, curr_sp, seed)]

    MSE.append([method, curr_sp, seed, mse_archetypes, mse_encoding, time])

quantitative_results = pd.DataFrame(MSE)
quantitative_results.columns = pd.Index(['method', 'class', 'seed', 'mse_at', 'mse_at_space', 'runtime'])
quantitative_results.loc[quantitative_results['class'] == 0, 'class'] = 'Rectangle'
quantitative_results.loc[quantitative_results['class'] == 1, 'class'] = 'Oval'
quantitative_results.loc[quantitative_results['class'] == 2, 'class'] = 'Heart'


# Grouped barplot of MSE on the AT space
g = sns.catplot(data=quantitative_results, x='method', hue='class', y='mse_at_space', kind='bar', legend=False)
g.set_yticklabels(fontsize=22)

g.set_xticklabels(['AAnet\n(ours)', 'PCHA', 'kernel PCHA', 'PCHA on AE', 'Javadi et al.', 'Chen et al.'],
                  rotation=45, ha='right', fontsize=20)
scprep.plot.utils.shift_ticklabels(g.ax.xaxis, dx=0.25)
g.ax.set_xlabel('')
g.ax.set_ylabel('MSE', fontsize=30)
g.ax.figure.set_size_inches((6,8))
g.show()


# Plotting archetypes

fig, axes = plt.subplots(4,7, figsize=(14,8))
axes = axes

i = -1
seed = 0
curr_sp = 2 # Heart

true_archetypes = []
true_archetypal_coords = []

for weight in np.eye(4):
    img, weights = get_new_digit(basic_sprites[curr_sp], weight)
    img = img.flatten()
    true_archetypes.append(img)
    true_archetypal_coords.append(weight)
true_archetypes = np.array(true_archetypes)
true_archetypal_coords = np.vstack(true_archetypal_coords)

titles = ['Ground\ntruth', 'AAnet\n(ours)', 'PCHA', 'Kernel\nPCHA', 'PCHA\non AE','Javadi', 'Chen']
for i, method in enumerate(['Ground truth', 'AAnet', 'PCHA', 'kernelPCHA', 'PCHA_on_AE', 'Javadi', 'Chen']):
    c_axes = axes[:,i]
    for idx_at in range(4):

        ax = c_axes[idx_at]

        if method == 'Ground truth':
            new_archetypes = true_archetypes
            ax.set_ylabel('AT{}'.format(idx_at + 1),fontsize=30)
        else:
            mse_archetypes, mse_encoding, new_archetypal_coords, new_archetypes,\
            time = results[(method, curr_sp, seed)]

        mse, idx = mse_permute(true_archetypes, new_archetypes)
        mse, _, idx = run_comparisons.calc_MSE(true_archetypes, new_archetypes)

        new_archetypes = new_archetypes[idx]

        ax.matshow(new_archetypes[idx_at].reshape((56,56)), cmap='Greys_r')
        ax.set_xticks([])
        ax.set_yticks([])

        if idx_at == 0:
            title = titles[i]
            ax.set_title(title, fontsize=30)
fig.tight_layout()
fig.subplots_adjust(wspace=-0.5, hspace=0.05)


## Plotting latent spaces
fig, axes = plt.subplots(4,7, figsize=(13,8))
i = -1

titles = ['Ground\ntruth', 'AAnet\n(ours)', 'PCHA', 'Kernel\nPCHA', 'PCHA\non AE','Javadi', 'Chen']
for i, method in enumerate(['Ground truth', 'AAnet', 'PCHA', 'kernelPCHA', 'PCHA_on_AE', 'Javadi', 'Chen']):
        c_axes = axes[:,i]
        if method == 'Ground truth':
            new_archetypal_coords = true_archetypal_coords
            new_archetypes = true_archetypes
            ax.set_title('AT{}'.format(idx_at + 1),fontsize=16)

        else:
            mse_archetypes, mse_encoding, new_archetypal_coords, new_archetypes,\
            time = results[(method, curr_sp, seed)]

        mse, _, idx = run_comparisons.calc_MSE(true_archetypes, new_archetypes)

        new_archetypes = new_archetypes[idx]
        new_archetypal_coords = new_archetypal_coords[:, idx]
        archetype_mds = MDS(2).fit_transform(new_archetypes)
        data_mds = new_archetypal_coords @ archetype_mds
        for idx_at in range(4):
            ax = c_axes[idx_at]
            if method == 'Ground truth':
                ax.set_ylabel('AT{}'.format(idx_at + 1),fontsize=30)
            cvec = true_archetypal_coords[:,idx_at]
            #c_idx = np.argsort(cvec)
            ax.scatter(data_mds[:,0], data_mds[:,1], s=1, c=cvec[:])
            ax.scatter(archetype_mds[:,0], archetype_mds[:,1], s=200, c='#ee3658', zorder=3)

            for j in range(archetype_mds.shape[0]):
                ax.text(archetype_mds[j,0], archetype_mds[j,1], j+1,
                    horizontalalignment='center', verticalalignment='center',
                    fontdict={'color': 'white','size':10,'weight':'bold'}, zorder=4)


            ax.set_xticks([])
            ax.set_yticks([])
            #ax.set_xlabel('MDS1')
            #ax.set_ylabel('MDS2')
            #ax.set_title('AT{} loading'.format(idx_at+1))

            if idx_at == 0:
                title = titles[i]
                ax.set_title(title, fontsize=30)


fig.tight_layout()
fig.subplots_adjust(wspace=0.02, hspace=0.02)
'''
