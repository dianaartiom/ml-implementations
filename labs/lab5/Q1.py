import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from machine_learning.data_processing import split_data, split_on_batches, split_on_folds
from machine_learning.regression import sigmoid_activation, regularized_loss, regularized_update, derivative
from machine_learning.steplength import stepsize_adagrad
from machine_learning.error import rmse

import pdb
from mpl_toolkits.mplot3d import Axes3D

                
def stochastic_gradient_desent_ada(X_train, X_test, y_train, y_test, alpha, epochs, _lambda, batch_size):
    initial_alpha = alpha
    
    beta = np.zeros((X_train.shape[1], 1))
    rmse_train, rmse_test = [], []

    history = np.zeros((X_train.shape[1], 1))
    
    for epoch in range(epochs):
        
        gradient = derivative(X_train, y_train, beta)
        alpha, history = stepsize_adagrad(gradient, history, initial_alpha)
        
        for (batchX, batchY) in split_on_batches(X_train, y_train, batch_size):
            
            beta = regularized_update(batchX, batchY, beta, alpha, _lambda)    

        rmse_train.append(rmse(y_train, sigmoid_activation(X_train, beta)))
        rmse_test.append(rmse(y_test, sigmoid_activation(X_test, beta)))

    return beta, rmse_train, rmse_test


bank = pd.read_csv('bank.csv', sep = ';')
#redwine_data = pd.read_csv('winequality-red.csv', sep = ";")
#whitewine_data = pd.read_csv('winequality-white.csv', sep = ";" )

# replacing all yes and no values to 1's and 0's
bank = bank.replace({'yes' : 1, 'no' : 0})

# remove all NA values from the dataset
bank.dropna(inplace = True)
bank = pd.get_dummies(bank)

X = bank.loc[:, bank.columns != 'y']
X = (X - X.mean()) / (X.max() - X.min())
X = np.c_[np.ones((X.shape[0])), X]

y = bank[['y']]

X_train, X_test, y_train, y_test = split_data(X, y)

learning_rate = 0.01
epochs = 10
lamda = 0.1
batch_size = 50

beta, rmse_train, rmse_test = stochastic_gradient_desent_ada(X_train, X_test, y_train, y_test, learning_rate, \
                                  epochs,lamda, batch_size)


    
plt.plot(rmse_train)
plt.plot(rmse_test)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

            
def folds_sgd(X, y, k, alpha, epochs, _lambda, batch_size):
    
    current_rmse = []
    for X_train, y_train, X_test, y_test in split_on_folds(X, y, k):

        beta, rmse_train, rmse_test = stochastic_gradient_desent_ada(X_train, X_test, y_train, y_test, alpha, epochs, _lambda, batch_size)

        current_rmse.append(rmse_test[-1])

    return current_rmse

k = 5
batch_size = 50

def gridSearch(X, y, k, alpha, epochs, __lambda, batch_size):

    for _alpha in alpha:
        for _lambda in __lambda:
            rmse = folds_sgd(X, y, k, _alpha, epochs, _lambda, batch_size)
            
            yield(_alpha, _lambda, np.array(rmse).mean())


def labelize(x, y, z):
    return "a:" + str(x) + "   l:" + str(y) + "   e:" + str(round(z, 2))


def plot_grid_search(X, y, k, alpha, epochs, _lambda, batch_size):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    _min = 100000
    min_alpha = 0
    min_lambda = 0
    
    for _a, _b, _c in gridSearch(X, y, k, alpha, epochs, _lambda, batch_size):
        ax.scatter(_a, _b, _c, label = labelize(_a, _b, _c))
        
        if _min > _c:
            min_alpha, min_lambda, _min = _a, _b, _c
            
    ax.set_xlabel('Alpha')
    ax.set_ylabel('Lambda')
    ax.set_zlabel('RMSE')
    ax.legend()
            
    plt.show()
    
    return min_alpha, min_lambda

alpha = [0.000001, 0.00001, 0.01]
_lambda =[0.1,0.03, 0.07]

best_alpha, best_lambda = plot_grid_search(X, y, k, alpha, epochs, _lambda, batch_size)

beta, rmse_train, rmse_test = stochastic_gradient_desent_ada(X_train, X_test, y_train, y_test, learning_rate, \
                                  epochs,lamda, batch_size)

plt.plot(rmse_train)
plt.plot(rmse_test)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
