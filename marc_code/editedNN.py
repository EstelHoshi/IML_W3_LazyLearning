# Edited methods NN

import time
import numpy as np
import utils
import kNN


def get_ENN_pp(x_train, y_train, k, p):
    # Perform kNN (edited=1: not including the analysed sample)
    kNearestNeighbour = kNN.kNNAlgorithm(k=k, r=p, X_train=x_train, X_test=x_train, y_train=y_train, edited=1)
    y_pred = kNearestNeighbour.max_voting()

    # Edition policy
    # Edit the training set by erasing the training samples not fulfilling y_pred == y_train
    edited_positions = np.where(y_pred != y_train)
    edited_x_train = np.delete(x_train, edited_positions, axis=0)
    edited_y_train = np.delete(y_train, edited_positions)
    print('reduction proportion: ', str(edited_x_train.shape[0]/x_train.shape[0]))

    return edited_x_train, edited_y_train


def apply_ENN(x_train_enn, x_train, y_train):
    d_type = 'float64'

    for i in range(x_train.shape[1] - 1):
        d_type = d_type + ',float64'

    x_train = np.asarray(x_train, order='C')
    x_train_enn = np.asarray(x_train_enn, order='C')

    idx = np.in1d(x_train.view(dtype=d_type).reshape(x_train.shape[0]),
                  x_train_enn.view(dtype=d_type).reshape(x_train_enn.shape[0]))

    edited_x_train = x_train[idx]
    edited_y_train = y_train[idx]

    return edited_x_train, edited_y_train


def ENNTh(x_train, y_train, k, p, th):
    # Perform kNN with Neighbourhood probability (edited=1)
    kNearestNeighbour = kNN.kNNAlgorithm(k=k, r=p, X_train=x_train, X_test=x_train, y_train=y_train, edited=1)
    y_pred, pr = kNearestNeighbour.stochastic_edited_pr()

    # Edition policy
    edited_positions = np.where((y_pred != y_train) | (pr < th))
    edited_x_train = np.delete(x_train, edited_positions, axis=0)
    edited_y_train = np.delete(y_train, edited_positions)

    return edited_x_train, edited_y_train