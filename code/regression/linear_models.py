import numpy as np


def least_squares(y, tx):
    """ Ordinary least squares """
    return np.linalg.inv(tx.T.dot(tx)).dot(tx.T).dot(y)


def ridge_regression(y, tx, lambda_):
    """ Ridge regression (OLS with L2-regularization) """
    return np.linalg.inv(tx.T.dot(tx) + lambda_ * (2 * y.shape[0]) * np.eye(tx.shape[1])).dot(tx.T).dot(y)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    """ Build polynomial features out of the given data, up to degree 'd' >= 1 """
    # At least degree 1
    if degree < 1:
        degree = 1
    
    tx = np.ones((x.shape[0], 1))
    for i in range(degree):
        tx = np.hstack((tx, x**(i+1)))
    return tx


def polynomial_regression(y, tx, lambda_, degree):
    x_poly = build_poly(tx, degree)
    return ridge_regression(y, x_poly, lambda_)
