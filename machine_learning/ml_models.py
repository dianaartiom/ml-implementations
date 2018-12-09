import numpy as np
import pandas as pd


def sigmoid(X, b):
    z = np.dot(X, b)
    return 1. / (np.power(np.e, -z) + 1)


def loglikelihood(X, y, b):
    h = sigmoid(X, b)
    
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
