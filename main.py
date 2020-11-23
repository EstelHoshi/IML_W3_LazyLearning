import click
from click import Choice
import numpy as np

import datasetsCBR
from classification import kNNAlgorithm, kNNAlgorithm_Estel
from reduction import reductionKNNAlgorithm


@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Perform PCA on a dataset and then reconstruct it
@cli.command('kNN')
@click.option('-d', '--dataset', nargs=2, type=(Choice(['kropt', 'satimage', 'credita']), Choice([str(i) for i in range(10)])), 
              default=('satimage', 0), help='[kropt|satimage|credita] [0,9]\n Dataset name, fold')
@click.option('-k', type=int, default=3, help='Value k for the nearest neighours to consider')
@click.option('-s', '--similarity', type=Choice(['minkowski1', 'minkowski2', 'XXX']), default='minkowski2', 
              help='Distance / similarity function')
@click.option('-p', '--policy', type=Choice(['majority', 'inverse', 'sheppard']), default='majority',
              help='Policy for deciding the solution of a query')
@click.option('-w', '--weighting', type=Choice(['equal', 'XXX']), default='equal',
              help='Method to weight features')
def kNN(dataset, k, similarity, policy, weighting):
    if dataset[0] == 'kropt':
        kNN_kropt(dataset[1], k, similarity, policy, weighting)

    elif dataset[0] == 'satimage':
        kNN_satimage(dataset[1], k, similarity, policy, weighting)

    elif dataset[0] == 'credita':
        kNN_credita(dataset[1], k, similarity, policy, weighting)


def kNN_kropt(i, k, similarity, policy, weighting):
    X_train, X_test, y_train, y_test = datasetsCBR.load_kropt(i)


def kNN_satimage(i, k, similarity, policy, weighting):
    X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(i)
    kNNy_test = kNNAlgorithm_Estel(X_train,y_train,X_test,k, similarity, policy, weighting)
    y_test = np.array(y_test).astype(int)
    acc = np.nansum((y_test-kNNy_test)/(y_test.astype(int)-kNNy_test))
    acc = 100-100*acc/len(y_test)
    print(acc)


def kNN_credita(i, k, similarity, policy, weighting):
    X_train, X_test, y_train, y_test = datasetsCBR.load_credita(i)
    print(X_train)
    print(y_train)


if __name__ == "__main__":
    cli()