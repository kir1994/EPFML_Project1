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

    return x[indices[:train_size]], y[indices[:train_size]], x[indices[train_size:]], y[indices[train_size:]]


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, compute_loss, regression, **kwargs):
    """Here should be some description"""
    loss_tr = []
    loss_te = []
    for cur_k in range(k):
        indices_te = k_indices[cur_k]
        indices_tr = [i not in indices_te for i in range(len(y))]

        x_tr, y_tr = x[indices_tr], y[indices_tr]
        x_te, y_te = x[indices_te], y[indices_te]

        weights = regression(y_tr, x_tr, **kwargs)
        loss_te.append(compute_loss(y_te, x_te, weights, **kwargs))
        loss_tr.append(compute_loss(y_tr, x_tr, weights, **kwargs))

    return np.mean(loss_tr), np.mean(loss_te)

