import numpy as np


def Minkowski(r, x, y):
    if r == 1:  # Manhattan Distance (L1 norm)
        d = np.abs(x[:, None] - y).sum(-1)
    elif r == 2:  # Euclidean Distance (L2 norm)
        d = np.sqrt(np.sum(x**2, axis=1)[:, np.newaxis] - 2*np.dot(x, y.T) + np.sum(y**2, axis=1)[np.newaxis, :])
    elif r == 'inf':  # Chebyshev distance (sup Linf norm)
        d = np.max(abs(x[:, None] - y), axis=2)
    else:
        print('Invalid r')

    return d


def mode(x):
    mode = np.empty((x.shape[0], 1))

    for i in range(x.shape[0]):
        unique, counts = np.unique(x[i, :], return_counts=True)
        max_idx = np.argmax(counts)  # in case of tie select the smallest one
        mode[i, 0] = unique[max_idx]

    return mode
