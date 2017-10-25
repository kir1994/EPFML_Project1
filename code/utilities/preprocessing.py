import numpy as np


def standard_scaler(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)


def standard_scaler_outlier(x, outlier):
    x_scaled = x.copy()
    means = []
    sds = []
    for i in range(x.shape[1]):
        ind_ok = x_scaled[:,i] != outlier
        ind_nok = x_scaled[:,i] == outlier
        mean = np.mean(x_scaled[ind_ok, i])
        sd = np.std(x_scaled[ind_ok, i])
        x_scaled[ind_nok,i] = mean
        x_scaled[:,i] -= mean
        x_scaled[:,i] /= sd
        means.append(mean)
        sds.append(sd)
    return x_scaled, means, sds