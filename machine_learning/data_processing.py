import numpy as np
import pandas as pd
from math import isnan

def split_data(X, y, proportion=0.8):
    '''
    splits data in test and training given the proportion
    
    Keyword arguments:
    X - numpy matrix - data
    target - numpy vector
    proportion = number between 0 and 1 = proportion for training and testing data set
    '''
    rand_select = np.random.rand(len(X)) <= proportion

    X_train = X[rand_select]
    X_test = X[~rand_select]
    
    y_train = y[rand_select]
    y_test = y[~rand_select]
    
    return X_train, X_test, y_train, y_test


def load_data(path, sep=' ', with_header=False):
    """
    loads data given the stored in path
    
    Keyword arguments:
    path - relative path to the file where data is stored
    sep - separator used in csv
    with_header - bianry value, checks if header should be loaded
    """
    if(with_header):
        return pd.read_csv(path, sep=sep)
    
    return pd.read_csv(path, sep=sep, header=None)

def hot_encode(df):
    return None

def check_nan(df, columns):
    count_nan = len(df) - df.count()
    x = False

    for i in range(len(count_nan)):
        if count_nan[i] != 0:
            x = True

    return x

def convert_to_binary(df, columns):
	_df = df
	for column in columns:
		_df[column] = pd.Series(np.where(_df[column].values == 'yes', 1, 0), _df.index)
	
	return _df


def split_on_batches(X, y, batchSize):
	for i in np.arange(0, X.shape[0], batchSize):
		yield X[i:i + batchSize], y[i:i + batchSize]

def split_on_folds(X, y, k):
    batchSize = int(round(len(y) / k))

    for j in np.arange(0, X.shape[0], batchSize):
        X_test = X[j:j + batchSize]
        y_test = y[j:j + batchSize]
        for i in np.arange(0, X.shape[0], batchSize):
            if i == j:
                continue
            else:
                yield X[i:i + batchSize], y[i:i + batchSize], X_test, y_test

