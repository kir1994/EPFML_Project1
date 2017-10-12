import numpy as np
from utilities.stochastic import sigmoid
from regression.linear_models import build_poly


def compute_mse_loss(y, tx, w, lambda_=0):
    """ Calculate the MSE loss """
    return np.sum((y - tx.dot(w)) ** 2) / (2. * y.shape[0]) + lambda_ * w.dot(w.T)


def compute_mse_loss_poly(y, tx, w, lambda_=0, degree=1):
    """ Calculate the MSE loss """
    x_poly = build_poly(tx, degree)
    return compute_mse_loss(y, x_poly, w, lambda_)


def compute_rmse_loss(y, tx, w, lambda_=0):
    """ Calculate the RMSE loss """
    return np.sqrt(2 * compute_mse_loss(y, tx, w, lambda_))


def compute_rmse_loss_poly(y, tx, w, lambda_=0, degree=1):
    """ Calculate the RMSE loss """
    return np.sqrt(2 * compute_mse_loss_poly(y, tx, w, lambda_, degree))


def compute_mae_loss(y, tx, w):
    """ Calculate the MAE loss """
    return np.sum(np.abs(y - tx.dot(w)) / (2. * y.shape[0]))


def compute_logistic_loss(y, tx, w, lambda_ = 0):
    """ Log-loss for logistic regression """
    return -np.sum(y * np.log(sigmoid(tx, w)) + (1 - y) * np.log(1 - sigmoid(tx, w))) / y.shape[0] + \
        lambda_ * w[np.newaxis, :].dot(w[:, np.newaxis])
