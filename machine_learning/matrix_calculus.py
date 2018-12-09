# -*- coding: utf-8 -*-
import numpy as np

def multiply_element_wise(X1, X2):
    """
    multiplies each element in X1 with the element in X2 at the same index

    Keyword arguments:
    X1 - numpy vector column
    X2 - numpy vector column
    """
    mul_ = []
    for i in range(len(X1)):
        mul_.append(X1[i] * X2[i])
        
    
    return np.array(mul_)
    