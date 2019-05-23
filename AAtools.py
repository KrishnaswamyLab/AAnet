# Plotting functions for AAnet
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_mnist(samples, shape=(28,28), n_cols=12):
    n = samples.shape[0]
    n_cols = np.minimum(n_cols,n)
    n_rows = np.ceil(n / n_cols).astype('int')
    fig = plt.figure(figsize=(n_cols*1, n_rows*1))
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(shape), cmap='Greys_r')

    return ax
def plot_frey(samples, n_cols=12):
    n = samples.shape[0]
    n_cols = np.minimum(n_cols,n)
    n_rows = np.ceil(n / n_cols).astype('int')
    fig = plt.figure(figsize=(n_cols*1.5*(20/28), n_rows*1.5))
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 20), cmap='Greys_r')

def plot_celeba(samples, n_cols=12, D=28):
    samples = samples + 1
    samples = samples / 2
    n = samples.shape[0]
    n_cols = np.minimum(n_cols,n)
    n_rows = np.ceil(n / n_cols).astype('int')
    fig = plt.figure(figsize=(n_cols*1.5, n_rows*1.5))
    gs = gridspec.GridSpec(n_rows, n_cols)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(D, D, 3))
