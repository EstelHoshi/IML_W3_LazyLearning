import numpy as np
import click
import datasetsCBR

@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Perform PCA on a dataset and then reconstruct it
@cli.command('kNN')
@click.option('-d',nargs = 2, type=(str,int),default=('satimage',0), help='Dataset name kropt | satimage | credita \n 0:9')
def kNN(d):
    if d[0] == 'kropt':
        kNN_kropt(d[1])

    elif d[0] == 'satimage':
        kNN_satimage(d[1])

    elif d[0] == 'credita':
        kNN_credita(d[1])

    else:
        raise ValueError('Unknown dataset {}'.format(d))

def kNN_kropt(i):
    X, y = datasetsCBR.load_kropt(i)


def kNN_satimage(i):
    X_train, y_train, X_test, y_test = datasetsCBR.load_satimage(i)
    print(X_train)
    print(y_train)

def kNN_credita(i):
    X, y = datasetsCBR.load_credita(i)


if __name__ == "__main__":
    cli()