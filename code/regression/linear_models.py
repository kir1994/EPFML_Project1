import numpy as np


def least_squares(y, tx):
    """ Ordinary least squares """
    return np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)


def ridge_regression(y, tx, lambda_):
    """ Ridge regression (OLS with L2-regularization) """
    return np.linalg.inv(tx.T.dot(tx) + lambda_ * (2 * y.shape[0]) * np.eye(tx.shape[1])).dot(tx.T).dot(y)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    return np.array([x ** j for j in range(degree + 1)]).T


def polynomial_regression(y, tx, lambda_, degree):
    x_poly = build_poly(tx, degree)
    return ridge_regression(y, x_poly, lambda_)
