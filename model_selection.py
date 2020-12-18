import numpy as np
import pandas as pd
import itertools
import time
from sklearn.metrics import accuracy_score
from classification import kNNAlgorithm
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
from joblib import Parallel, delayed


def GridSearchCV(cv_splits, param_grid, n_proc=1):
    combs = generate_combinations(param_grid)

    if n_proc == 1:
        results = list(map(_bootstrap_cv, [(cv_splits, c) for c in combs]))
    else:
        with ThreadPool(processes=n_proc) as pool:
            results = pool.map(_bootstrap_cv, [(cv_splits, c) for c in combs])

    # create df with results
    history = pd.DataFrame(results, columns=['accuracy', 'efficiency'])
    cols = list(param_grid.keys())
    values = [list(c.values()) for c in combs]
    history[cols] = values
    return history


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


def _bootstrap_cv(args):
        cv, params = args
        acc, eff = cross_validate(cv, **params)

        # print(round(acc, 4), params)
        return acc, eff


def cross_validate(cv, **params):
    acc = 0
    eff = 0

    for X_train, X_test, y_train, y_test in cv:
        t0 = time.time()
        y_pred = kNNAlgorithm(X_test, X_train, y_train, **params)
        eff += time.time() - t0
        acc += accuracy_score(y_test, y_pred)

    k = len(cv)
    return acc / k, eff / k
