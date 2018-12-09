import numpy as np


def sigmoid_activation(X, b):
    z = np.dot(X, b)

    return 1.0 / (1 + np.exp(-z))


def regularized_loss(data, labels, beta, _lambda):
    p = sigmoid_activation(data, beta)
    cost = -labels * np.log(p) - (1 - labels) * np.log(1 - p) \
    - (_lambda * (sum(beta**2)))
    return round(cost.mean().values[0], 4)


def regularized_update(data, labels, beta, alpha, _lambda):
    p = sigmoid_activation(data, beta)
    return beta - alpha * (-2 * np.dot(data.T, labels - p) + (2 * _lambda) * beta)

def derivative(X, y, b):
    y_hat = sigmoid_activation(X, b)
    
    return np.dot(X.T, y - y_hat)