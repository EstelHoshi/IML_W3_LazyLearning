import numpy as np
import pandas as pd
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool


def _sanitize(X):
    return X.values if isinstance(X, pd.DataFrame) else X


def kNNAlgorithm(q, X, y, k, distance='2-norm', policy='majority'):
    X = _sanitize(X)
    y = _sanitize(y)
    q = _sanitize(q)

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
        knn_indices = np.argsort(dists)[:k]

        knn_dists = dists[knn_indices]
        X_knn = X[knn_indices]
        y_knn = y[knn_indices]
        return vote_func(qi, X_knn, y_knn, knn_dists)
        
    with ThreadPool(processes=cpu_count()) as pool:
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
    inv_dist = 1. / dists

    win_vote = -1
    win_label = 0
    for label in np.unique(y).flat:
        vote = np.sum(inv_dist[y == label])
        
        if vote > win_vote:
            win_vote, win_label = vote, label

    return win_label


def _vote_sheppard(q, X, y, dists):
    exp_dist = np.exp(-dists)

    win_vote = -1
    win_label = 0
    for label in np.unique(y).flat:
        vote = np.sum(exp_dist[y == label])
        
        if vote > win_vote:
            win_vote, win_label = vote, label

    return win_label
