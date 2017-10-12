import numpy as np


def standard_scaler(x):
    return (x - np.mean(x, axis=0)) / np.std(x, axis=0)

