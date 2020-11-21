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

    return df_train, df_train_label, df_test, df_test_label


def load_credita(i):

    return i, i


