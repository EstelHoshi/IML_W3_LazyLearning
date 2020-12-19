import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool


def _sanitize(X):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    return X


def kNNAlgorithm(q, X, y, k, distance='2-norm', policy='majority', edit=False, n_proc=1):
    X = _sanitize(X)
    y = _sanitize(y)
    q = _sanitize(q)

    # make it 2d array if only 1 query point was passed, to be able to iterate
    q = q[None, :] if isinstance(q, np.ndarray) and q.ndim == 1 else q

    dist_func = {
        '1-norm': lambda a, b: minkowski(a, b, 1),
        '2-norm': lambda a, b: minkowski(a, b, 2),
        'chebyshev': lambda a, b: minkowski(a, b, np.inf),
    }[distance]

    vote_func = {
        'majority': _vote_majority,
        'inverse': _vote_inverse,
        'sheppard': _vote_sheppard
    }[policy]
    

    dist = dist_func(X, q)
    if edit:
        dist[np.eye(dist.shape[0], dtype=bool)] = 1000

    knn_idx = np.argsort(dist, axis=0)[:k].T
    y_pred = np.empty(q.shape[0])

    for i in range(q.shape[0]):
        y_pred[i] = vote_func(q[i], X[knn_idx[i]], y[knn_idx[i]], dist[knn_idx[i]])

    return y_pred
    
def _dist_1norm(a, b):
    return minkowski(a, b, 1)
def _dist_2norm():
    return minkowski(a, b, 2)
def _dist_chebyshev():
    return minkowski(a, b, np.inf)

def minkowski(x, y, r):
    if r == 1:  # Manhattan Distance (L1 norm)
        d = np.sum(np.abs(x[:, None] - y), axis=2)

    elif r == 2:  # Euclidean Distance (L2 norm)
        d = np.sqrt(np.sum(x**2, axis=1)[:, np.newaxis] - 2*np.dot(x, y.T) + np.sum(y**2, axis=1))

    elif r == np.inf:  # Chebyshev distance (sup Linf norm)
        d = np.max(np.abs(x[:, None] - y), axis=2)

    return d


def _vote_majority(q, X, y, *args):
    values, count = np.unique(y, return_counts=True)
    return values[np.argmax(count)]


def _vote_inverse(q, X, y, dists):
    inv_dist = np.divide(1., dists)
    return _vote_distance(q, X, y, inv_dist)


def _vote_sheppard(q, X, y, dists):
    exp_dist = np.exp(-dists)
    return _vote_distance(q, X, y, exp_dist)
    

def _vote_distance(q, X, y, dists):
    win_vote = -1
    win_label = 0
    for label in np.unique(y).flat:
        vote = np.sum(dists[y == label])
        
        if vote > win_vote:
            win_vote, win_label = vote, label

    return win_label
