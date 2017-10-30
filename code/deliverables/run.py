# -*- coding: utf-8 -*-
"""
Load the datasets, train a model, and create a Kaggle submission for the first 
Machine Learning project

Authors: Kirill IVANOV, Matthias RAMIREZ, Nicolas TALABOT
"""

### Import modules and datasets
from proj1_helpers import load_csv_data, predict_labels, create_csv_submission
from implementations import least_squares
from utilities import split_data, preprocess_data

y_train, x_train, ids_train = load_csv_data("train.csv")
y_test, x_test, ids_test = load_csv_data("test.csv")

# Parameters
seed   = 3
degree = 11
ratio  = 0.66

# Learn the model
tx, x_mean, x_std = preprocess_data(x_train, degree)
x_tr, y_tr, x_te, y_te = split_data(tx, y_train, ratio, seed)
w, loss_tr = least_squares(y_tr, x_tr)

# Create a Kaggle submission
x_kaggle,_,_ = preprocess_data(x_test, degree, compute_mean_std=False, \
                               x_mean=x_mean, x_std=x_std)
y_pred = predict_labels(w, x_kaggle)
create_csv_submission(ids_test, y_pred, "run_submission.csv")
