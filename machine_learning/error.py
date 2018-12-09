#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 13:52:50 2018

@author: dianaartiom
"""
import numpy as np
import math
from math import sqrt


def rmse(y, y_hat):
    """
    Root mean squared error
    
    Keyword arguments:
    y - real values of target
    y_hat - predicted values of target
    """
    return sqrt(np.sum(pow((y-y_hat),2))/len(y))
