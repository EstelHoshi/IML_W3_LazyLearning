import numpy as np
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool


def _sanitize(X):
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    return X


def kNNAlgorithm(q, X, y, k, distance='2-norm', policy='majority', offset=0, n_proc=1):
    X = _sanitize(X)
    y = _sanitize(y)
    q = _sanitize(q)

    # make it 2d array if only 1 query point was passed, to be able to iterate
    q = q[None, :] if isinstance(q, np.ndarray) and q.ndim == 1 else q

    dist_func = {
        '1-norm': _dist_1norm,
        '2-norm': _dist_2norm,
        'chebyshev': _dist_chebyshev
    }[distance]

    vote_func = {
        'majority': _vote_majority,
        'inverse': _vote_inverse,
        'sheppard': _vote_sheppard
    }[policy]
    
    def single_knn(qi):
        dists = dist_func(X, qi)
        knn_indices = np.argsort(dists)[offset:k+offset]
        return vote_func(qi, X[knn_indices], y[knn_indices], dists[knn_indices])

    if n_proc == 1: # sequential
        return np.array([single_knn(qi) for qi in q])
    else: # parallel
        with ThreadPool(processes=n_proc) as pool:
            y_pred = pool.map(single_knn, q)
        return np.array(y_pred)


def _dist_1norm(a, b):
    return np.linalg.norm(b - a, ord=1, axis=1)


def _dist_2norm(a, b):
    return np.linalg.norm(b - a, ord=2, axis=1)


def _dist_chebyshev(a, b):
    return np.max(np.abs(b - a), axis=1)


def _vote_majority(q, X, y, *args):
    values, count = np.unique(y, return_counts=True)
    return values[np.argmax(count)]


def _vote_inverse(q, X, y, dists):
    dists[dists == 0] = 1e-9    # avoid divisions by 0
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
