#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:55:11 2018

@author: dianaartiom
"""
# from numpy import inv
import numpy as np
from math import sqrt

def solve_SLE(A, b):
    return inv(A).dot(b)

def learn_linreg_NormEq(X, y):
    """
    returns weights vector in linear regression using normal equations
    
    Keyword arguments:
    X = data set
    y = target
    """
    A =  X.T.dot(X)
    b = X.T.dot(y)
    
    return solve_SLE(A, b)

def lin_reg(X, b):
    """
    retuns predicted y vector (y_hat)
    
    X - data set
    b - parameter weights
    
    """
    return X.dot(b)

def y_hat(data, beta):
    return np.dot(data,beta)

def loss_function(data, labels, beta, lamda):
    yhat = y_hat(data, beta)
    result = sqrt(np.sum(pow((labels-yhat),2))) + (lamda * (sum(beta**2)))
    return result

def derivative(data, labels, beta, lamda):
    p = y_hat(data, beta)
    return (-2 * np.dot(data.T, labels - p) + (2 * lamda) * beta)
