import os
import glob
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from ReliefF import ReliefF


def load_kropt(i):
    return i, i


def load_satimage(i):
    train_path = os.path.join('datasetsCBR/satimage', 'satimage.fold.00000'+str(i)+'.train.arff')
    test_path = os.path.join('datasetsCBR/satimage', 'satimage.fold.00000'+str(i)+'.test.arff')

    train_subset = loadarff(train_path)
    test_subset = loadarff(test_path)

    df_train = pd.DataFrame(train_subset[0])
    df_test = pd.DataFrame(test_subset[0])

    df_train_label = pd.DataFrame(df_train["clase"])
    df_test_label = pd.DataFrame(df_test["clase"])

    return df_train, df_train_label, df_test, df_test_label


def load_credita(weighting=None, **extra_kwargs):
    cv_splits = []

    # preprocess the first fold keeping statistics for next folds
    train_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.000000.train.arff')
    test_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.000000.test.arff')

    df_train = pd.DataFrame(loadarff(train_path)[0])
    df_test = pd.DataFrame(loadarff(test_path)[0])

    X = df_train.append(df_test)
    y = X.pop('class')

    y_label_encoder = LabelEncoder()
    y = y_label_encoder.fit_transform(y)

    # fill missing numerical values
    means = X.mean()
    X.fillna(means, inplace=True)

    # fill missing categorical values
    categ_cols = X.select_dtypes(include=['category', object]).columns
    modes = X[categ_cols].mode()
    for col in categ_cols:
        X[col].replace(b'?', modes[col][0], inplace=True)

    # standarize numerical features
    num_cols = X.select_dtypes(include=['number']).columns
    mm_scaler = MinMaxScaler()
    X[num_cols] = mm_scaler.fit_transform(X[num_cols])

    # use one transformer per feature to preserve its name in the generated features
    # since new feature names are based on the transformer's name
    transformers = [(col, OneHotEncoder(drop='first'), [col]) for col in categ_cols]
    col_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_arr = col_transformer.fit_transform(X)

    X = pd.DataFrame(X_arr, columns=col_transformer.get_feature_names())

    p = len(df_train)
    X_train, X_test, y_train, y_test = X[:p], X[p:], y[:p], y[p:]

    # feature selection
    if weighting == 'information_gain':
        discrete_features = np.ones(len(X_train.columns), dtype=np.bool)
        discrete_features[:-len(num_cols)] = False
        weights = mutual_info_classif(X_train, y_train, discrete_features=discrete_features)
        
        # normalize weights
        min_ = np.min(weights)
        max_ = np.max(weights)
        weights = (weights - min_) / (max_ - min_)

        # apply weights to features
        X_train *= weights
        X_test *= weights

    elif weighting == 'relief':
        n_neighbors = extra_kwargs.get('n_neighbors', 200)
        relief = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=X.shape[1])
        relief.fit(X_train.values, y_train)
        weights = relief.feature_scores

        # normalize weights
        min_ = np.min(weights)
        max_ = np.max(weights)
        weights = (weights - min_) / (max_ - min_)
        weights[weights == 0] = 1e-6

        # apply weights to features
        X_train *= weights
        X_test *= weights

    cv_splits.append((X_train, X_test, y_train, y_test))

    # preprocess rest of folds
    for i in range(1, 10):
        train_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.00000{str(i)}.train.arff')
        test_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.00000{str(i)}.test.arff')

        df_train = pd.DataFrame(loadarff(train_path)[0])
        df_test = pd.DataFrame(loadarff(test_path)[0])

        X = df_train.append(df_test)
        y = X.pop('class')

        y = y_label_encoder.transform(y)

        # fill missing numerical values
        X.fillna(means, inplace=True)

        # fill missing categorical values
        for col in categ_cols:
            X[col].replace(b'?', modes[col][0], inplace=True)

        # normalize numerical features
        X[num_cols] = mm_scaler.transform(X[num_cols])

        # one hot encode
        X_arr = col_transformer.transform(X)
        X = pd.DataFrame(X_arr, columns=col_transformer.get_feature_names())

        p = len(df_train)
        X_train, X_test, y_train, y_test = X[:p], X[p:], y[:p], y[p:]

        # feature selection
        if weighting == 'information_gain':
            discrete_features = np.ones(len(X_train.columns), dtype=np.bool)
            discrete_features[:-len(num_cols)] = False
            weights = mutual_info_classif(X_train, y_train, discrete_features=discrete_features)
            
            # normalize weights
            min_ = np.min(weights)
            max_ = np.max(weights)
            weights = (weights - min_) / (max_ - min_)

            # apply weights to features
            X_train *= weights
            X_test *= weights

        elif weighting == 'relief':
            n_neighbors = extra_kwargs.get('n_neighbors', 200)
            relief = ReliefF(n_neighbors=n_neighbors, n_features_to_keep=X.shape[1])
            relief.fit(X_train.values, y_train)
            weights = relief.feature_scores

            # normalize weights
            min_ = np.min(weights)
            max_ = np.max(weights)
            weights = (weights - min_) / (max_ - min_)
            weights[weights == 0] = 1e-6

            # apply weights to features
            X_train *= weights
            X_test *= weights

        cv_splits.append((X_train, X_test, y_train, y_test))
    
    return cv_splits


