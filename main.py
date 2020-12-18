import click
from click import Choice
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
from tqdm import tqdm

import datasetsCBR
from classification import kNNAlgorithm
from reduction import reductionKNNAlgorithm
from model_selection import GridSearchCV, cross_validate


K_FOLDS = 10


@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Run kNN with specific parameters
@cli.command('kNN')
@click.option('-d', '--dataset', type=Choice(['kropt', 'satimage', 'credita']), default='satimage',
              help='Dataset name')
@click.option('-k', type=int, default=3, help='Value k for the nearest neighours to consider')
@click.option('-s', '--similarity', type=Choice(['1-norm', '2-norm', 'chebyshev']), default='2-norm', 
              help='Distance / similarity function')
@click.option('-p', '--policy', type=Choice(['majority', 'inverse', 'sheppard']), default='majority',
              help='Policy for deciding the solution of a query')
@click.option('-w', '--weighting', type=Choice(['uniform', 'information_gain', 'relief']), default='uniform',
              help='Method to weight features')
@click.option('-r', '--reduction', type=Choice(['no', 'drop2', 'drop3', 'XXX', 'YYY']), default='no',
              help='Method to reduce the number of instances')
def kNN(dataset, k, similarity, policy, weighting, reduction):
    if dataset == 'credita':
        cv_splits = datasetsCBR.load_credita(weighting=weighting)

    elif dataset == 'satimage':
        cv_splits = []
        for j in range(K_FOLDS):
            X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(j)
            X_train.pop('clase')
            y_train = y_train.astype(np.int64).values.ravel()
            X_test.pop('clase')
            y_test = y_test.astype(np.int64).values.ravel()
            cv_splits.append((X_train, X_test, y_train, y_test))
        # no: 0.9102, drop2: 0.8965, drop3: 0.8925

    # perform reduction
    if reduction != 'no':
        # reduce each fold separately
        saved_inst_count = 0
        
        t0 = time.time()
        for i in tqdm(range(len(cv_splits))):   # progress bar
            X_train, X_test, y_train, y_test = cv_splits[i]
            X_redc, y_redc = reductionKNNAlgorithm(X_train, y_train, k=k, algorithm=reduction,
                                                   distance=similarity, policy=policy)
            cv_splits[i] = X_redc, X_test, y_redc, y_test
            saved_inst_count += X_redc.shape[0]

        redc_eff = (time.time() - t0) / len(cv_splits)
        print('reduction efficiency: {}s'.format(round(redc_eff, 2)))

        avg_redc = saved_inst_count / len(cv_splits)
        n = X_train.shape[0] + X_test.shape[0]
        print('reduction in storage: {}/{} ({}%)'.format(round(avg_redc, 2), n, round(avg_redc * 100 / n, 2)))
        

    stats = cross_validate(cv_splits, k=k, distance=similarity, policy=policy)

    print('accuracy:', round(stats[0], 6))
    print(f'efficiency: {round(stats[1], 6)}s')


# -----------------------------------------------------------------------------------------
# Perform a grid search over all possible combination of parameters for kNN
@cli.command('gridsearch')
@click.option('-d', '--dataset', type=Choice(['satimage', 'credita']), default='credita',
              help='Dataset name')
@click.option('-o', '--out', type=str, default=None, help='Output file name')
@click.option('-cv', type=str, default='yes', help='Whether to compute the average over k-folds or '
              'consider each fold as a separate dataset')
def gridsearch(dataset, out, cv):
    if dataset == 'credita':
        cv_splits_un = datasetsCBR.load_credita(weighting=None)
        cv_splits_ig = datasetsCBR.load_credita(weighting='information_gain')
        cv_splits_rf = datasetsCBR.load_credita(weighting='relief')

        cv_splits_all = [cv_splits_un, cv_splits_ig, cv_splits_rf]

    elif dataset == 'satimage':
        cv_splits = []
        for j in range(K_FOLDS):
            X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(j)
            X_train.pop('clase')
            y_train = y_train.astype(np.int64).values.ravel()
            X_test.pop('clase')
            y_test = y_test.astype(np.int64).values.ravel()
            cv_splits.append((X_train, X_test, y_train, y_test))

        cv_splits_all += cv_splits * 3  # temporal fix

    if cv == 'no':
        # flatten, squeeze to perform gridsearch on each fold separately
        cv_splits_all = [[split] for cv_splits in cv_splits_all for split in cv_splits]


    param_grid = {
        'k': [1, 3, 5, 7],
        'distance': ['1-norm', '2-norm', 'chebyshev'],
        'policy': ['majority', 'inverse', 'sheppard']
    }

    # run gridsearch and accumulate results
    history = pd.DataFrame()

    with tqdm(total=len(cv_splits_all)) as pbar: # progress bar

        for split in cv_splits_all:
            h = GridSearchCV(split, param_grid)
            history = history.append(h, ignore_index=True)
            pbar.update()

    # tag results with corresponding weighting methods
    m = len(history) // 3
    history.loc[:m*1, 'weighting'] = 'uniform'
    history.loc[m*1:m*2, 'weighting'] = 'information_gain'
    history.loc[m*2:, 'weighting'] = 'relief'

    if cv == 'no':
        # tag fold number: first 36 to 0, next 36 to 1, ...
        c = 4 * 3 * 3   # number of results per one fold
        history['fold'] = dataset + '-' +  ((history.index // c) % K_FOLDS).astype(str)

    # print best
    print('\nbest:')
    print(history.sort_values('accuracy').iloc[-1])

    # save df
    if out is not None:
        if not out.endswith('.csv'):
            out += '.csv'
        history.to_csv(out, index=False)


if __name__ == "__main__":
    cli()