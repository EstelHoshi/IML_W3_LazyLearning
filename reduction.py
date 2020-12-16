import numpy as np
from multiprocessing.dummy import Pool as ThreadPool
from classification import (kNNAlgorithm, _sanitize, _dist_1norm, _dist_2norm,
                            _dist_chebyshev, _vote_majority, _vote_inverse,
                            _vote_sheppard)


def reductionKNNAlgorithm(X, y, algorithm='drop2', **kwargs):
    algorithm = algorithm.lower()

    if algorithm == 'drop2':
        return DROP(X, y, v=2, **kwargs)
    elif algorithm == 'drop3':
        return DROP(X, y, v=3, **kwargs)


def DROP(X, y, distance='2-norm', v=2, n_proc=2, **knn_kwargs):
    T = _sanitize(X)
    y = _sanitize(y)

    dist_func = {
        '1-norm': _dist_1norm,
        '2-norm': _dist_2norm,
        'chebyshev': _dist_chebyshev
    }[distance]

    knn_kwargs['distance'] = distance
    knn_kwargs['n_proc'] = 1
    k = knn_kwargs['k']
    
    if v == 3: # DROP3
        T, y = ENN(T, y, **knn_kwargs)

    n = T.shape[0]
    S = np.ones(n, dtype=np.bool)
    A = [list() for _ in range(n)]

    def single_knn_min_enemy_dist(idx):
        dists = dist_func(T, T[idx])
        sorted_i = np.argsort(dists)
        knn_result = sorted_i[1:k+2]
        min_enemy_dist = dists[sorted_i][y[sorted_i] != y[idx]][0] #  np.min(dists[y != y[idx]])
        return knn_result, min_enemy_dist

    def single_update_assoc(a):
        dists = dist_func(T, T[a])
        sorted_i = np.argsort(dists)
        ini = int(S[a]) # if is still present, ignore distance to itself
        knn[a, :] = sorted_i[S[sorted_i]][ini:k+1+ini] # k+1 dist to points still present
        A[knn[a, -1]].append(a)

    with ThreadPool(processes=n_proc) as pool:
        # compute k+1 nn for each sample (list of nn) and distance to nearest enemy
        res = pool.map(single_knn_min_enemy_dist, range(n))
        knn = np.array([r[0] for r in res])

        # create list of associates
        for i, indices in enumerate(knn):
            for j in indices:
                A[j].append(i)

        # compute order of removal with distances to nearest enemies
        enemy_dists = [r[1] for r in res]
        order_of_removal = np.argsort(enemy_dists)[::-1]

        for i in order_of_removal:
            # count associates of i correctly classified
            assoc = A[i]
            with_mask = np.unique(knn[assoc, :k])
            y_pred_with = kNNAlgorithm(T[assoc], T[with_mask], y[with_mask], **knn_kwargs)
            with_ = np.count_nonzero(y[assoc] == y_pred_with)

            # count associates of i correctly classified without i
            assoc_knn = knn[assoc, :]   # this time get k+1 nn from each assoc
            without_mask = np.unique(assoc_knn[assoc_knn != i]) # and remove i
            y_pred_without = kNNAlgorithm(T[assoc], T[without_mask], y[without_mask], **knn_kwargs)
            without_ = np.count_nonzero(y[assoc] == y_pred_without)

            if without_ >= with_:
                # remove i and update associates with their next nn
                S[i] = False
                pool.map(single_update_assoc, assoc)
            
    return T[S], y[S]


def ENN(X, y, **knn_kwargs):
    n = X.shape[0]
    mask = np.ones(n, dtype=np.bool)
    y_pred = np.empty_like(y)
    for i in range(n):
        mask[i] = False
        y_pred[i] = kNNAlgorithm(X[i], X[mask], y[mask], **knn_kwargs)[0]
        mask[i] = True

    return X[y_pred == y], y[y_pred == y] #, y_pred != y