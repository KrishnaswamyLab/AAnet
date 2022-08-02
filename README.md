# Archetypal Analysis network (AAnet)

## Quick start

* [Guided tutorial in Python](https://nbviewer.jupyter.org/github/KrishnaswamyLab/AAnet/blob/master/AAnet_torch/simulated_data_example/AAnet_vs_other_methods_on_tetrahedron.ipynb)


## Introduction

AAnet is a tool for scalable Archetypal Analysis of large and potentially non-linear datasets. A full description of the algorithm is available in our manuscript on ArXiv.

[D. van Dijk, D. Burkhardt, et al. Finding Archetypal Spaces for Data Using Neural Networks. 2019. arXiv](https://arxiv.org/abs/1901.09078)

![alt text](https://github.com/KrishnaswamyLab/AAnet/blob/master/AAnet.png)

Archetypal analysis is a data decomposition method that describes each observation in a dataset as a convex combination of "pure types" or archetypes. Existing methods for archetypal analysis work well when a linear relationship exists between the feature space and the archetypal space. However, such methods are not applicable to systems where the feature space is generated non-linearly from the combination of archetypes, such as in biological systems or image transformations. Here, we propose a reformulation of the problem such that the goal is to learn a non-linear transformation of the data into a latent archetypal space. To solve this problem, we introduce Archetypal Analysis network (AAnet), which is a deep neural network framework for learning and generating from a latent archetypal representation of data. We demonstrate state-of-the-art recovery of ground-truth archetypes in non-linear data domains, show AAnet can generative from data geometry rather than from data density, and use AAnet to identify biologically meaningful archetypes in single-cell gene expression data.

## Using this repository

Currently, AAnet is not a full Python package, but all of the code you need to run the algorithm is in this repo. For convenience, we've organized code into folders for AAnet proper and for other algorithms to which we compared AAnet in our manuscript.

The most updated version of AAnet (08/01/2022) is implemented in torch. 

### File list
* `AAnet_torch/models`- Includes most recent iteration of AAnet model
* `AAnet_torch/data` - Includes helper functions for generating curved simplices for testing
* `AAnet_torch/simulated_data_example` - Includes Jupyter notebook and helper functiosn to run AAnet
