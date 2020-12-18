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
    if dataset == 'kropt':
        kNN_kropt(k, similarity, policy, weighting, reduction)

    elif dataset == 'satimage':
        kNN_satimage(k, similarity, policy, weighting, reduction)

    elif dataset == 'credita':
        kNN_credita(k, similarity, policy, weighting, reduction)


def kNN_kropt(k, similarity, policy, weighting):
    X_train, X_test, y_train, y_test = datasetsCBR.load_kropt(i)


def kNN_satimage(k, similarity, policy, weighting):
    X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(i)
    print(X_train)
    print(y_train)


def kNN_credita(k, similarity, policy, weighting, reduction):
    cv_splits = datasetsCBR.load_credita(weighting=weighting)
    # cv_splits = []
    # for j in range(K_FOLDS):
    #     X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(j)
    #     X_train.pop('clase')
    #     y_train = y_train.astype(np.int64).values.ravel()
    #     X_test.pop('clase')
    #     y_test = y_test.astype(np.int64).values.ravel()
    #     cv_splits.append((X_train, X_test, y_train, y_test))
    # no: 0.9102, drop2: 0.8965, drop3: 0.8925

    # perform reduction
    if reduction != 'no':
        redc_count = 0
        t0 = time.time()
        for i in range(len(cv_splits)):
            X_train, X_test, y_train, y_test = cv_splits[i]
            X_redc, y_redc = reductionKNNAlgorithm(X_train, y_train, k=k, algorithm=reduction,
                                                   distance=similarity, policy=policy)
            cv_splits[i] = X_redc, X_test, y_redc, y_test
            redc_count += X_redc.shape[0]

        redc_eff = (time.time() - t0) / len(cv_splits)
        print('reduction efficiency: {}s'.format(round(redc_eff, 2)))

        avg_redc = redc_count / len(cv_splits)
        n = X_train.shape[0] + X_test.shape[0]
        print('reduction storage: {}/{} ({}%)'.format(round(avg_redc, 2), n, round(avg_redc * 100 / n, 2)))
        
    stats = cross_validate(cv_splits, k=k, distance=similarity, policy=policy)

    print('accuracy:', round(stats[0], 6))
    print(f'efficiency: {round(stats[1], 6)}s')


# -----------------------------------------------------------------------------------------
# Perform a grid search over all possible combination of parameters for kNN
@cli.command('gridsearch')
@click.option('-d', '--dataset', type=Choice(['kropt', 'satimage', 'credita']), default='satimage',
              help='Dataset name')
@click.option('-o', '--out', type=str, default=None,
              help='Output file name')
def gridsearch(dataset, out):
    if dataset == 'credita':
        cv_splits_un = datasetsCBR.load_credita(weighting=None)
        cv_splits_ig = datasetsCBR.load_credita(weighting='information_gain')
        cv_splits_rf = datasetsCBR.load_credita(weighting='relief')

    elif dataset == 'satimage':
        cv_splits = []
        for j in range(K_FOLDS):
            X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(j)
            X_train.pop('clase')
            y_train = y_train.astype(np.int64).values.ravel()
            X_test.pop('clase')
            y_test = y_test.astype(np.int64).values.ravel()
            cv_splits.append((X_train, X_test, y_train, y_test))

    param_grid = {
        'k': [1, 3, 5, 7],
        'distance': ['1-norm', '2-norm', 'chebyshev'],
        'policy': ['majority', 'inverse', 'sheppard']
    }

    history = GridSearchCV(cv_splits_un, param_grid)
    history['weighting'] = 'uniform'

    h = GridSearchCV(cv_splits_ig, param_grid)
    h['weighting'] = 'information_gain'
    history = history.append(h)

    h = GridSearchCV(cv_splits_rf, param_grid)
    h['weighting'] = 'relief'
    history = history.append(h)

    history = history.sort_values('accuracy')
    print('\nbest:')
    print(history.iloc[-1])

    if out is not None:
        if not out.endswith('.csv'):
            out += '.csv'
        history.to_csv(out, index=False)



# -----------------------------------------------------------------------------------------
# Perform a grid search over all possible combination of parameters for kNN
@cli.command('param-analysis')
@click.option('-d', '--dataset', type=Choice(['kropt', 'satimage', 'credita']), default='satimage',
              help='Dataset name')
@click.option('-o', '--out', type=str, default=None,
              help='Output file name')
@click.option('-cv', type=str, default='yes',
              help='Output file name')
def param_analysis(dataset, out, cv):
    if dataset == 'credita':
        cv_splits_un = datasetsCBR.load_credita(weighting=None)
        cv_splits_ig = datasetsCBR.load_credita(weighting='information_gain')
        cv_splits_rf = datasetsCBR.load_credita(weighting='relief')

        cv_splits_all = [cv_splits_un, cv_splits_ig, cv_splits_rf]

    elif dataset == 'satimage':
        # cv_splits = []
        # for j in range(K_FOLDS):
        #     X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(j)
        #     X_train.pop('clase')
        #     y_train = y_train.astype(np.int64).values.ravel()
        #     X_test.pop('clase')
        #     y_test = y_test.astype(np.int64).values.ravel()
        #     cv_splits.append((X_train, X_test, y_train, y_test))

        # cv_splits_all += cv_splits * 3
        pass

    if cv == 'no':
        cv_splits_all = [[split] for cv_splits in cv_splits_all for split in cv_splits]

    param_grid = {
        'k': [1, 3, 5, 7],
        'distance': ['1-norm', '2-norm', 'chebyshev'],
        'policy': ['majority', 'inverse', 'sheppard']
    }


    history = pd.DataFrame()

    with tqdm(total=len(cv_splits_all)) as pbar:
        for split in cv_splits_all:
            h = GridSearchCV(split, param_grid)
            history = history.append(h, ignore_index=True)
            pbar.update()

    c = 4 * 3 * 3   # 36, number of combinations of parameters
    m = len(history) // 3

    history.loc[:m*1, 'weighting'] = 'uniform'
    history.loc[m*1:m*2, 'weighting'] = 'information_gain'
    history.loc[m*2:m*3, 'weighting'] = 'relief'

    if cv == 'no':
        # tag fold number first 36 to 0, next 36 to 1, ...
        history['fold'] += dataset + '-' +  ((history.index // c) % K_FOLDS).astype(str)

    # print best
    history = history.sort_values('accuracy')
    print('\nbest:')
    print(history.iloc[-1])

    # save df
    if out is not None:
        if not out.endswith('.csv'):
            out += '.csv'
        history.to_csv(out, index=False)

if __name__ == "__main__":
    cli()