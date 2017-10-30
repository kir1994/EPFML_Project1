# -*- coding: utf-8 -*-
"""
Load the datasets, train a model, and create a Kaggle submission for the first 
Machine Learning project

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""

### Import modules and datasets
import numpy as np
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from implementations import least_squares
from utilities import split_data, build_poly

y_train, x_train, ids_train = load_csv_data("train.csv")
y_test, x_test, ids_test = load_csv_data("test.csv")


### Pre-process the training data
# Parameters
seed   = 3
degree = 11
ratio  = 0.66

# Build polynomial features (without PRI_jet_num, and discard the bias column)
x = build_poly(np.delete(x_train, 22, axis=1), degree)[:,1:]

# Create a mask for the missing values (=False)
x_train_mask = (np.delete(x_train, 22, axis=1) != -999.)
x_mask = x_train_mask
for i in range(degree-1):
    x_mask = np.c_[x_mask, x_train_mask]

# Loop on the features, compute the mean/std without the missing values
x_mean = np.zeros(x.shape[1])
x_std  = np.zeros(x.shape[1])
for i in range(x.shape[1]):
    feature_values = x[x_mask[:, i], i]
    x_mean[i] = feature_values.mean()
    x_std[i]  = feature_values.std()

# Normalize and center the data
x = (x - x_mean) / x_std

# Set the missing values to 0 (= the mean)
x[np.invert(x_mask)] = 0.

# Add the bias column of "1's"
tx = np.c_[np.ones(x.shape[0]), x]

# Add the Binary features
tx = np.c_[tx, x_train[:,0] == -999, x_train[:,22] == 1, x_train[:,22] == 2 , \
           x_train[:,22] == 3]


### Split the data
x_tr, y_tr, x_te, y_te = split_data(tx, y_train, ratio, seed)


### Train the model
# Least-saqures
w, loss_tr = least_squares(y_tr, x_tr)


### Create a Kaggle submission
## Send the test set to the same space as the train set
# Build polynomial features (without PRI_jet_num)
x_kaggle = build_poly(np.delete(x_test, 22, axis=1), degree)

# Normalize and center the data (+ keep the "1's" column)
x_kaggle = np.c_[x_kaggle[:,0], (x_kaggle[:,1:] - x_mean) / x_std]

# Set to 0 (=the mean) the missing values
x_test_mask = (np.delete(x_test, 22, axis=1) != -999.0)
x_kaggle_mask = np.c_[np.ones(x_kaggle.shape[0], dtype=bool), x_test_mask]
for i in range(degree-1):
    x_kaggle_mask = np.c_[x_kaggle_mask, x_test_mask]
x_kaggle[np.invert(x_kaggle_mask)] = 0.

# Boolean features
x_kaggle = np.c_[x_kaggle, x_test[:,0] == -999, x_test[:,22] == 1, \
                 x_test[:,22] == 2 , x_test[:,22] == 3]

## Create a submission file
# Predict the labels
y_pred = predict_labels(w, x_kaggle)

# Create a sumbission file to be uploaded to the Kaggle competition
create_csv_submission(ids_test, y_pred, "run_submission.csv")
