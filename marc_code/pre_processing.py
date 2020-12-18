from scipy.io import arff
import pandas as pd
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import sklearn_relief as relief
from ReliefF import ReliefF
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2


def obtain_pp(train_fold, evaluation_fold, weighting):
    data_train = pd.DataFrame(arff.loadarff(train_fold)[0]).dropna().to_numpy()
    data_eval = pd.DataFrame(arff.loadarff(evaluation_fold)[0]).dropna().to_numpy()

    data_complete = np.concatenate((data_train, data_eval))

    # Perform pre-processing for the complete data

    # Pre-Processing

    # Convert label features into numerical with Label Encoder
    label_encoder = []
    le_values = np.zeros_like(data_complete)

    for i in range(data_complete.shape[1]):
        le = LabelEncoder()
        label_encoder.append(le.fit(data_complete[:, i]))
        le_values[:, i] = le.transform(data_complete[:, i])

    scaler = MinMaxScaler()
    scaler.fit(le_values[:, :data_complete.shape[1] - 1])

    # Obtain scaled data for weighting purposes
    X = scaler.transform(le_values[:, :data_complete.shape[1] - 1])
    Y = le_values[:, data_complete.shape[1] - 1].astype(int)

    if weighting == 'uniform':
        weights = np.ones(X.shape[1])
        X = X * weights
    elif weighting == 'mutual_info':
        raw_weights = mutual_info_classif(X, Y)
        max_w = raw_weights.max()
        min_w = raw_weights.min()
        weights = raw_weights / max_w
        #weights = (raw_weights - min_w)/(max_w - min_w)
        X = X * weights
    elif weighting == 'chi2':
        raw_weights = chi2(X, Y)[0]
        max_w = raw_weights.max()  # consider min-max instead of only max
        min_w = raw_weights.min()
        weights = raw_weights / max_w
        #weights = (raw_weights - min_w) / (max_w - min_w)
        X = X * weights
    elif weighting == 'relieff':
        relieff = ReliefF(n_neighbors=100, n_features_to_keep=X.shape[1])
        relieff.fit(X, Y)
        raw_weights = abs(relieff.feature_scores)
        max_w = raw_weights.max()  # consider min-max instead of only max
        min_w = raw_weights.min()
        #weights = (raw_weights - min_w) / (max_w - min_w)
        weights = raw_weights / max_w
        X = X * weights
    else:  # if incorrect input, uniform weighting
        weights = np.ones(X.shape[1])
        X = X * weights

    return scaler, label_encoder, weights, X, Y


def apply_pp(dataset_fold, le, scaler, weights):
    # Read input dataset
    dataset_path = os.path.join('datasetsCBR', dataset_fold.split('.')[0])
    dataset = os.path.join(dataset_path, dataset_fold)

    data = pd.DataFrame(arff.loadarff(dataset)[0]).dropna().to_numpy()

    # Pre-Processing

    # Convert label features into numerical with Label Encoder
    label_encoder = np.zeros_like(data)

    for i in range(data.shape[1]):
        label_encoder[:, i] = le[i].transform(data[:, i])

    X = scaler.transform(label_encoder[:, :data.shape[1]-1])
    Y = label_encoder[:, data.shape[1]-1].astype(int)

    # Apply weights
    X = X * weights

    return X, Y