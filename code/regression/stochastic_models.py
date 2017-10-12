from utilities.stochastic import stochastic_gradient_descent, sigmoid
from regression.loss import compute_mse_loss, compute_mae_loss, compute_logistic_loss
from numpy import newaxis


def compute_mse_gradient(y, tx, w):
    """ Gradient for MSE loss """
    return -tx.T.dot(y - tx.dot(w)) / y.shape[0]


def compute_mae_gradient(y, tx, w):
    """ Gradient for MAE loss """
    e = y - tx.dot(w)
    e[e > 0] = 1
    e[e < 0] = -1
    return -tx.T.dot(e) / (2 * y.shape[0])


def compute_logistic_gradient(y, tx, w, lambda_=0):
    """ Gradient for logistic loss """
    return (y - sigmoid(tx, w))[newaxis, :].dot(tx) + lambda_ / y.shape[0] * w[newaxis, :]


def least_squares_gd(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_mse_loss, compute_mse_gradient)


def least_squares_sgd(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_mse_loss, compute_mse_gradient, batch_size = 1)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_logistic_loss, compute_logistic_gradient)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_logistic_loss, compute_logistic_gradient, lambda_ = lambda_)
