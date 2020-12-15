import os
import glob
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict
from sklearn.feature_selection import mutual_info_classif


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


def load_credita(feature_selection=None):
    cv_splits = []

    # preprocess first fold keeping statistics for next folds
    train_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.000000.train.arff')
    test_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.000000.test.arff')

    X_train = pd.DataFrame(loadarff(train_path)[0])
    X_test = pd.DataFrame(loadarff(test_path)[0])

    X = X_train.append(X_test)
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

    # feature selection
    if feature_selection == 'information_gain':
        discrete_features = np.ones(len(X.columns), dtype=np.bool)
        discrete_features[:-len(num_cols)] = False
        weights = mutual_info_classif(X, y, discrete_features=discrete_features)
        
        # normalize weights
        min_ = np.min(weights)
        max_ = np.max(weights)
        weights = (weights - min_) / (max_ - min_)

        # apply weights to features
        X *= weights
    else:
        weights = None

    p = len(X_train)
    cv_splits.append((X[:p], X[p:], y[:p], y[p:]))


    # preprocess rest of folds
    for i in range(1, 10):
        train_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.00000{str(i)}.train.arff')
        test_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.00000{str(i)}.test.arff')

        X_train = pd.DataFrame(loadarff(train_path)[0])
        X_test = pd.DataFrame(loadarff(test_path)[0])

        X = X_train.append(X_test)
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

        # feature selection
        if weights is not None:
            X *= weights

        p = len(X_train)
        cv_splits.append((X[:p], X[p:], y[:p], y[p:]))
    
    return cv_splits


