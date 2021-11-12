# Copyright Krishnaswamy Lab 2021
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import torch
import warnings

def plot_latent_space(model, data, c=None, ax=None):
    '''Method for visualizing the latent archetypal space. Algorithm is:
    1. MDS is performed on the archetypes in the feature space to provide
       a frame for the data.
    2. Points represented as a mixture of archetypes are interpolated between
       the calculated coordinates for each at
    This could also be achieved by running MDS on the latent space + archetypes. This
    interpolation approach is faster and yeilds similar results.'''

        # Encode the data into the latent space
    data_latent = model.encode(torch.Tensor(data).type(torch.float))
    # Translate to barycentric coordinates
    data_barycentric = model.euclidean_to_barycentric(data_latent).detach().numpy()

    # Get archetypes in feature space
    archetypes_features = model.get_archetypes_data().detach().numpy()
    embedding = MDS(n_components=2)

    #Filter MDS warning
    message =  "The MDS API has changed. ``fit`` now constructs an" \
               " dissimilarity matrix from data. To use a custom " \
               "dissimilarity matrix, set " \
               "``dissimilarity='precomputed'``."
    warnings.filterwarnings("ignore", message=message)
    archetypes_MDS = embedding.fit_transform(archetypes_features)

    # Linearly interpolate between the archetypal coordiantes
    data_MDS = data_barycentric @ archetypes_MDS

    if ax is None:
        fig, ax = plt.subplots(1, figsize=(8,6))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('MDS1')
    ax.set_ylabel('MDS2')
    ax.scatter(data_MDS[:,0], data_MDS[:,1], c=c, s=1, alpha=0.5)
    ax.scatter(archetypes_MDS[:,0], archetypes_MDS[:,1], s=200, c='r', zorder=3)
    for i in range(archetypes_MDS.shape[0]):
        ax.text(archetypes_MDS[i,0],
                archetypes_MDS[i,1],
                '{}'.format(i+1),
                horizontalalignment='center',
                verticalalignment='center',
                fontdict={'color': 'white','size':10,'weight':'bold'},
                zorder=4)
    return ax
