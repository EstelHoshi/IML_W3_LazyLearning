import click
from click import Choice
import numpy as np
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time

import datasetsCBR
from classification import kNNAlgorithm
from reduction import reductionKNNAlgorithm
from model_selection import GridSearchCV


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
@click.option('-w', '--weighting', type=Choice(['equal', 'XXX', 'YYY']), default='equal',
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
    cv_splits = datasetsCBR.load_credita(feature_selection=weighting)
    # cv_splits = []
    # for j in range(K_FOLDS):
    #     X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(j)
    #     X_train.pop('clase')
    #     y_train = y_train.astype(np.int64).values.ravel()
    #     X_test.pop('clase')
    #     y_test = y_test.astype(np.int64).values.ravel()
    #     cv_splits.append((X_train, X_test, y_train, y_test))
    # # no: 0.9102, drop2: 0.8965, drop3: 0.8925

    # perform reduction
    if reduction != 'no':
        t0 = time.time()
        for i in range(len(cv_splits)):
            X_train, X_test, y_train, y_test = cv_splits[i]
            X_redc, y_redc = reductionKNNAlgorithm(X_train, y_train, k=k, algorithm=reduction,
                                                   distance=similarity, policy=policy)
            cv_splits[i] = X_redc, X_test, y_redc, y_test
        print('reduce time:', round(time.time() - t0, 2), 's')
        
    sum_accuracy = 0
    for X_train, X_test, y_train, y_test in cv_splits:
        y_pred = kNNAlgorithm(X_test, X_train, y_train, k, distance=similarity, policy=policy)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        sum_accuracy += accuracy
        print(round(accuracy, 4))

    print('mean accuracy:', round(sum_accuracy / K_FOLDS, 4))


# -----------------------------------------------------------------------------------------
# Perform a grid search over all possible combination of parameters for kNN
@cli.command('gridsearch')
@click.option('-d', '--dataset', type=Choice(['kropt', 'satimage', 'credita']), default='satimage',
              help='Dataset name')
def gridsearch(dataset):
    cv_splits = datasetsCBR.load_credita()
    # cv_splits = []
    # for j in range(K_FOLDS):
    #     X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(6)
    #     X_train.pop('clase')
    #     y_train = y_train.astype(np.int64).values.ravel()
    #     X_test.pop('clase')
    #     y_test = y_test.astype(np.int64).values.ravel()
    #     cv_splits.append((X_train, X_test, y_train, y_test))

    param_grid = {
        'k': [1, 3, 5, 7],
        'distance': ['1-norm', '2-norm', 'chebyshev'],
        'policy': ['majority', 'inverse', 'sheppard']
    }
    t0 = time.time()
    best = GridSearchCV(cv_splits, param_grid)
    print('best:', round(best[0], 6), best[1])
    print(round(time.time() - t0, 2), 's')

if __name__ == "__main__":
    cli()