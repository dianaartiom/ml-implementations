#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:54:49 2018

@author: dianaartiom
"""

from linear_regression import learn_linreg_NormEq, lin_reg
from error import rmse

def search_forward(X_train, X_test, y_train, y_test):
    
    """
    implements search forward algorithm, returns features selected
    
    Keyword arguments:
    X_train - training data set 
    y_train - training target
    X_test - testing data set
    y_test - testing target
    """
    beta0 = learn_linreg_NormEq(X_train, y_train)
    y_hat = lin_reg(X_test, beta0)
    
    e_all_best = 2000000
    V = []
    v_best = 1

    all_features = list(range(len(X_test.T)))

    while(v_best != 100):
        v_best = 100
        e_best = e_all_best
        
        
        for v in (list(set(all_features) - set(V))):

            current_feature = []
            current_feature.append(v)

            V_prime = V + current_feature
            
            beta_ = learn_linreg_NormEq(X_train[:, V_prime], y_train)
            y_hat = lin_reg(X_test[:, V_prime], beta_)
            e = rmse(y_test, y_hat)

            if e < e_best:
                e_best = e
                v_best = v
                
        if e_best < e_all_best:
            V.append(v_best)
            e_all_best = e_best
        
    return V