# -*- coding: utf-8 -*-
"""
A few useful functions for run.py

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""
import numpy as np

def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8
    you will have 80% of your data set dedicated to training
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)

    indices = np.random.permutation(y.shape[0])
    train_size = int(y.shape[0] * ratio)

    return x[indices[:train_size]], y[indices[:train_size]],  \
            x[indices[train_size:]], y[indices[train_size:]]

def build_poly(x, d=1):
    """ Build polynomial features out of the given data, up to degree 'd' >= 1 """
    # At least degree 1
    if d < 1:
        d = 1
    
    tx = np.ones(x.shape[0])
    for i in range(d):
        tx = np.c_[tx, x**(i+1)]
    return tx