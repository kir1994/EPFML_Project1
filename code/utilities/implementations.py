import numpy as np

##### Losses #####
def compute_mse_loss(y, tx, w):
    """ Calculate the MSE loss """
    return np.sum((y - tx.dot(w))**2) / (2. * y.shape[0])

def compute_mae_loss(y, tx, w):
    """ Calculate the MAE loss """
    return np.sum(np.abs(y - tx.dot(w)) / (2. * y.shape[0]))

def compute_logistic_loss(y, tx, w, lambda_ = 0):
	""" Log-loss for logistic regression """
	return np.sum(np.log(1. + np.exp(tx.dot(w))) - y * tx.dot(w)) + \
             lambda_ * w[np.newaxis, :].dot(w[:, np.newaxis])[0, 0]

##### Gradients #####
def compute_mse_gradient(y, tx, w):
    """ Gradient for MSE loss """
    return - tx.T.dot(y - tx.dot(w)) / y.shape[0]
	
def compute_mae_gradient(y, tx, w):
    """ (sub-)Gradient for MAE loss """
    e = y - tx.dot(w)
    e[e > 0] = 1
    e[e < 0] = -1
    return -tx.T.dot(e) / (2 * y.shape[0])

def sigmoid(z):
	""" Logistic function """
	return 1. / (1. + np.exp(-z))

def compute_logistic_gradient(y, tx, w, lambda_=0):
    """ Gradient for logistic loss """
    return tx.T.dot(sigmoid(tx.dot(w)) - y) + 2. * lambda_ * w

##### Gradient descent / SGD #####
def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, \
                                compute_loss, compute_gradient, \
                                batch_size = -1, **kwargs):
    """ Stochastic gradient descent algorithm """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    y_cur = y
    tx_cur = tx
    for n_iter in range(max_iters):
        # If batch size = -1, do normal gradient descent
        if batch_size > 0:
            indices = np.random.permutation(y.shape[0])
            tx_cur = tx[indices[:batch_size], :]
            y_cur = y[indices[:batch_size]]
        
        # Compute gradient and loss		
        loss = compute_loss(y_cur, tx_cur, w, **kwargs)
        grad = compute_gradient(y_cur, tx_cur, w, **kwargs)
		
        # Update w by gradient
        w = w - gamma * grad
        
        # Store w and loss
        ws.append(w)
        losses.append(loss)
        if n_iter % 100 == 0:
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format( \
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return ws[-1], losses[-1]

############################################
########## IMPLEMENTED ML METHODS ##########

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using gradient descent """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, \
                                       compute_mse_loss, compute_mse_gradient)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """ Linear regression using stochastic gradient descent (mini-batch-size 1) """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, \
                                       compute_mse_loss, compute_mse_gradient, \
                                       batch_size = 1)

def least_squares(y, tx):
    """ Least squares regression using normal equations """
    try:
        w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError:
        w = np.linalg.lstsq(tx.T.dot(tx), tx.T.dot(y))[0]
    loss = compute_mse_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """ Ridge regression using normal equations """
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * \
                        (2. * y.shape[0]) * np.eye(tx.shape[1]), tx.T.dot(y))
    loss = compute_mse_loss(y, tx, w) + \
           lambda_ * w[np.newaxis, :].dot(w[:, np.newaxis])[0, 0]
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic regression using gradient descent """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, \
                                       compute_logistic_loss, \
                                       compute_logistic_gradient)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression using gradient descent """
    return stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, \
                                       compute_logistic_loss, \
                                       compute_logistic_gradient, lambda_ = lambda_)

############################################
############################################