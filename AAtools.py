import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_mnist(samples, nc=12):
    n = samples.shape[0]
    nc = np.minimum(nc,n)
    nr = np.ceil(n / nc).astype('int')
    fig = plt.figure(figsize=(nc*1, nr*1))
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

def plot_frey(samples, nc=12):
    n = samples.shape[0]
    nc = np.minimum(nc,n)
    nr = np.ceil(n / nc).astype('int')
    fig = plt.figure(figsize=(nc*1.5*(20/28), nr*1.5))
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 20), cmap='Greys_r')

def plot_celeba(samples, nc=12, D=28):
    samples = samples + 1
    samples = samples / 2
    n = samples.shape[0]
    nc = np.minimum(nc,n)
    nr = np.ceil(n / nc).astype('int')
    fig = plt.figure(figsize=(nc*1.5, nr*1.5))
    gs = gridspec.GridSpec(nr, nc)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(D, D, 3))