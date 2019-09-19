# Archetypal Analysis network (AAnet)

## Quick start

* [Guided tutorial in Python](https://nbviewer.jupyter.org/github/KrishnaswamyLab/AAnet/blob/master/notebooks/AAnet_tutorial_MNIST.ipynb)


## Introduction

AAnet is a tool for scalable Archetypal Analysis of large and potentially non-linear datasets. A full description of the algorithm is available in our preprint on ArXiv.

[D. van Dijk, D. Burkhardt, et al. Finding Archetypal Spaces for Data Using Neural Networks. 2019. arXiv](https://arxiv.org/abs/1901.09078)

![alt text](https://github.com/KrishnaswamyLab/AAnet/blob/master/img/AAnet.png)

Archetypal analysis is a data decomposition method that describes each observation in a dataset as a convex combination of "pure types" or archetypes. Existing methods for archetypal analysis work well when a linear relationship exists between the feature space and the archetypal space. However, such methods are not applicable to systems where the feature space is generated non-linearly from the combination of archetypes, such as in biological systems or image transformations. Here, we propose a reformulation of the problem such that the goal is to learn a non-linear transformation of the data into a latent archetypal space. To solve this problem, we introduce Archetypal Analysis network (AAnet), which is a deep neural network framework for learning and generating from a latent archetypal representation of data. We demonstrate state-of-the-art recovery of ground-truth archetypes in non-linear data domains, show AAnet can generative from data geometry rather than from data density, and use AAnet to identify biologically meaningful archetypes in single-cell gene expression data.

## Using this repository

Currently, AAnet is not a full Python package, but all of the code you need to run the algorithm is in this repo. For convenience, we've organized code into folders for AAnet proper and for other algorithms to which we compared AAnet in our manuscript.

### File list

`AAnet code/`
* `AAnet.py` - Python implementation of AAnet. Requires `network.py`
* `AAtools.py` - Plotting functions that are helpful when running AAnet
* `network.py` - Base code for an AE encoder and decoder

`comparison_code/` - Code for running all AA methods and for rerunning the dSprites experiment
* `run_comparisons.py` - Includes a method, runAA, for running at AA methods in the paper
* `image_translation_comparisons.py` - Generates data for dSprites comparison and runs all 6 methods. Includes functions for saving and visualizing results in Fig. 4.2.
* `furthest_sum.py` - Helper function for Javadi et al.
* `Javadi.py` - We had to manually convert the Javadi et al. method to Python3.
* `PCHA.py` - Obtained from https://github.com/ulfaslak/py_pcha

`notebooks/` - Contains Jupyter notebook tutorial(s)
* `AAnet_tutorial_MNIST.ipynb` - A Jupyter notebook with a tutorial for running AAnet

### Getting help

If you have any questions about AAnet, please feel free to raise an Issue on GitHub. This will be the fastest way to get help. You can also email David (david.vandijk@yale.edu) or Daniel (daniel.burkhardt@yale.edu) with specific questions.
