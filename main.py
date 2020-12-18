import click
from click import Choice
import numpy as np

import datasetsCBR
from classification import kNNAlgorithm, kNNAlgorithm_Estel
from reduction import reductionKNNAlgorithm, reductionKNNAlgorithm_Estel


@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Perform PCA on a dataset and then reconstruct it
@cli.command('kNN')
#@click.option('-d', '--dataset', nargs=2, type=(Choice(['kropt', 'satimage', 'credita']), Choice([str(i) for i in range(10)])),
#              default=('satimage', 0), help='[kropt|satimage|credita] [0,9]\n Dataset name, fold')
@click.option('-d', '--dataset', type = Choice(['kropt', 'satimage', 'credita']), default = 'satimage',
                help = '[kropt|satimage|credita]')
@click.option('-k', type=int, default=3, help='Value k for the nearest neighours to consider')
@click.option('-s', '--similarity', type=Choice(['minkowski1', 'minkowski2', 'chebyshev']), default='minkowski1',
              help='Distance / similarity function')
@click.option('-p', '--policy', type=Choice(['majority', 'inverse', 'sheppard']), default='majority',
              help='Policy for deciding the solution of a query')
@click.option('-w', '--weighting', type=Choice(['equal', 'SelectKBest','ReliefF']), default='ReliefF',
              help='Method to weight features')
def kNN(dataset, k, similarity, policy, weighting):
    if dataset[0] == 'kropt':
        kNN_kropt(dataset[1], k, similarity, policy, weighting)

    elif dataset == 'satimage':
        kNN_satimage(k, similarity, policy, weighting)
        print("HOLA")

    elif dataset[0] == 'credita':
        kNN_credita(dataset[1], k, similarity, policy, weighting)


def kNN_kropt(i, k, similarity, policy, weighting):
    X_train, X_test, y_train, y_test = datasetsCBR.load_kropt(i)


def kNN_satimage(k, similarity, policy, weighting):
#    X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(i)
#    kNNy_test = kNNAlgorithm_Estel(X_train.to_numpy(),y_train.to_numpy(),X_test.to_numpy(),k, similarity, policy, weighting)

#    y_test = np.array(y_test).astype(int)
#    acc = np.nansum((y_test-kNNy_test)/(y_test.astype(int)-kNNy_test))
#    acc = 100-100*acc/len(y_test)
#    print(acc)
#    print(np.shape(y_train.to_numpy()))

    accuracy = []
    for i in range(10):
        X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(i)
        kNNy_test = kNNAlgorithm_Estel(X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), k, similarity, policy,weighting)
        y_test = np.array(y_test).astype(int)
        acc = np.nansum((y_test-kNNy_test)/(y_test.astype(int)-kNNy_test))
        acc = 100-100*acc/len(y_test)
        accuracy.append(acc)

    print(accuracy)
    avg_acc = np.sum(accuracy)/(len(accuracy))
    print(avg_acc)


def kNN_credita(i, k, similarity, policy, weighting):
    X_train, X_test, y_train, y_test = datasetsCBR.load_credita(i)
    print(X_train)
    print(y_train)


@cli.command('reductionkNN')
@click.option('-d', '--dataset', nargs=2, type=(Choice(['kropt', 'satimage', 'credita']), Choice([str(i) for i in range(10)])),
              default=('satimage', 0), help='[kropt|satimage|credita] [0,9]\n Dataset name, fold')
@click.option('-k', type=int, default=7, help='Value k for the nearest neighours to consider')
@click.option('-s', '--similarity', type=Choice(['minkowski1', 'minkowski2', 'chebyshev']), default='minkowski2',
              help='Distance / similarity function')
@click.option('-p', '--policy', type=Choice(['majority', 'inverse', 'sheppard']), default='majority',
              help='Policy for deciding the solution of a query')
@click.option('-w', '--weighting', type=Choice(['equal', 'SelectKBest']), default='equal',
              help='Method to weight features')

def reductionkNN(dataset, k, similarity, policy, weighting):
    if dataset[0] == 'kropt':
        reductionkNN_kropt(dataset[1], k, similarity, policy, weighting)

    elif dataset[0] == 'satimage':
        reductionkNN_satimage(dataset[1], k, similarity, policy, weighting)

    elif dataset[0] == 'credita':
        reductionkNN_credita(dataset[1], k, similarity, policy, weighting)

def reductionkNN_kropt(i, k, similarity, policy, weighting):
    pass
def reductionkNN_satimage(i, k, similarity, policy, weighting):
    X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(i)
    redkNNy_test = reductionKNNAlgorithm_Estel(X_train.to_numpy(),y_train.to_numpy(),X_test.to_numpy(),k, similarity, policy, weighting)
    y_test = np.array(y_test).astype(int)
    acc = np.nansum((y_test-redkNNy_test)/(y_test.astype(int)-redkNNy_test))
    acc = 100-100*acc/len(y_test)
    print(acc)


    #y_test = np.array(y_test).astype(int)
    #print(np.shape(y_test))
    #print(y_test)
    #acc = np.nansum((y_test-kNNy_test)/(y_test-kNNy_test))
    #acc = 100-100*acc/len(y_test)
    #print(acc)
def reductionkNN_credita(i, k, similarity, policy, weighting):
    pass
if __name__ == "__main__":
    cli()