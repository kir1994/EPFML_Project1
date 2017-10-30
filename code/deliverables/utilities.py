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

def standardize_missing(x, x_missing_mask):
    """ Standardize the data and set the missing values to 0(=the mean) """
    # Create empty arrays to store means and stds
    x_mean = np.zeros(x.shape[1])
    x_std  = np.zeros(x.shape[1])

    # Loop on the features, compute the mean/std without the missing values
    for i in range(x.shape[1]):
        feature_values = x[x_missing_mask[:, i], i]
        x_mean[i] = feature_values.mean()
        x_std[i]  = feature_values.std()
    
    # Normalize and center the data
    x_norm = (x - x_mean) / x_std
    # Set the missing values to 0 (= the mean)
    x_norm[np.invert(x_missing_mask)] = 0.
    return x_norm, x_mean, x_std

def preprocess_data(x_base, degree=1,\
                    compute_mean_std = True, x_mean = None, x_std = None):
    """ Pre-process the given data by constructing polynomial features,
    standardize the results, set the missing values to 0,
    and finally add binary features (cf. report for more informations) """
    # Build polynomial features (without PRI_jet_num)
    x = build_poly(np.delete(x_base, 22, axis=1), degree)
    
    # Create a mask for the missing values (missing=False)
    x_base_mask = (np.delete(x_base, 22, axis=1) != -999.)
    x_mask = x_base_mask
    for i in range(degree-1):
        x_mask = np.c_[x_mask, x_base_mask]
    
    # Standardize the features, and set the missing values to 0(=the mean)
    if compute_mean_std:
        tx, x_mean, x_std = standardize_missing(x[:,1:], x_mask)
        tx = np.c_[np.ones(x.shape[0]), tx]
    else: # Re-use existing means and stds
        tx = np.c_[x[:,0], (x[:,1:] - x_mean) / x_std]
        tx_mask = np.c_[np.ones(x.shape[0], dtype=bool), x_mask]
        tx[np.invert(tx_mask)] = 0.
    
    # Add the binary features
    tx = np.c_[tx, x_base[:,0] == -999, x_base[:,22] == 1, x_base[:,22] == 2 , \
               x_base[:,22] == 3]
    return tx, x_mean, x_std
