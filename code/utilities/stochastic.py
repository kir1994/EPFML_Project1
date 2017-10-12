import numpy as np


def stochastic_gradient_descent(y, tx, initial_w, max_iters, gamma, compute_loss, compute_gradient, batch_size=-1,**kwargs):
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
        loss = compute_loss(y_cur, tx_cur, w, kwargs)
        grad = compute_gradient(y_cur, tx_cur, w, kwargs)

        # Update w by gradient
        w = w - gamma * grad

        # Store w and loss
        ws.append(w)
        losses.append(loss)
        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return ws[-1], losses[-1]


def sigmoid(tx, w):
    """ Logistic function """
    return 1 / (1 + np.exp(-tx.dot(w)))