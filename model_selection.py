import numpy as np
import itertools
from sklearn.metrics import accuracy_score
from classification import kNNAlgorithm
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from joblib import Parallel, delayed


def GridSearchCV(cv_splits, param_grid, n_proc=2):
    combs = generate_combinations(param_grid)

    with Pool(processes=n_proc) as pool:
        results = pool.map(cross_validate_knn, [(cv_splits, c) for c in combs])
    # results = list(map(cross_validate_knn, [(cv_splits, c) for c in combs]))

    best_i = np.argmax(results)
    return results[best_i], combs[best_i]


def generate_combinations(param_grid):
    generator = itertools.product(*param_grid.values())

    keys = list(param_grid.keys())
    combs = []
    for comb in generator:
        params = {}
        for i, k in enumerate(keys):
            params[k] = comb[i]
        combs.append(params)
    return combs


def cross_validate_knn(args):
        cv_splits, params = args
        results = []
        for X_train, X_test, y_train, y_test in cv_splits:
            y_pred = kNNAlgorithm(X_test, X_train, y_train, **params)
            results.append(accuracy_score(y_test, y_pred))

        accuracy = np.mean(results)
        print(round(accuracy, 4), params)
        return accuracy