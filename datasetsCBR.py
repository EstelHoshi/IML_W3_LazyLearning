import os
import glob
import numpy as np
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from collections import defaultdict


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

    del df_train["clase"]
    del df_test["clase"]
    return df_train, df_train_label, df_test, df_test_label


def load_credita(i):
    train_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.00000{str(i)}.train.arff')
    test_path = os.path.join('datasetsCBR', 'credit-a', f'credit-a.fold.00000{str(i)}.test.arff')

    df_train = pd.DataFrame(loadarff(train_path)[0])
    df_test = pd.DataFrame(loadarff(test_path)[0])

    y_train = df_train.pop('class')
    y_test = df_test.pop('class')

    X_train = df_train
    X_test = df_test

    # Preprocess train split and use the same params,
    # encoders and transformers for the test split

    y_label_encoder = LabelEncoder()
    y_train = y_label_encoder.fit_transform(y_train)
    y_test = y_label_encoder.transform(y_test)

    # fill missing numerical values
    mean = X_train.mean()
    X_train.fillna(mean, inplace=True)
    X_test.fillna(mean, inplace=True)

    # fill missing categorical values
    categ_cols = X_train.select_dtypes(include=['category', object]).columns
    for col in categ_cols:
        mode = X_train[col].mode()[0]
        X_train[col].replace(b'?', mode, inplace=True)
        X_test[col].replace(b'?', mode, inplace=True)

    # standarize numerical features
    num_cols = X_train.select_dtypes(include=['number']).columns
    mm_scaler = MinMaxScaler()
    X_train[num_cols] = mm_scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = mm_scaler.transform(X_test[num_cols])

    # use one transformer per feature to preserve its name in the generated features
    # since new feature names are based on the transformer's name
    transformers = [(col, OneHotEncoder(drop='first'), [col]) for col in categ_cols]
    col_transformer = ColumnTransformer(transformers, remainder='passthrough')
    X_train_arr = col_transformer.fit_transform(X_train)
    X_test_arr = col_transformer.transform(X_test)

    X_train = pd.DataFrame(X_train_arr, columns=col_transformer.get_feature_names())
    X_test = pd.DataFrame(X_test_arr, columns=col_transformer.get_feature_names())

    return X_train, X_test, y_train, y_test


