import numpy as np

def stepsize_adagrad(gradient, history, initial_alpha):
    history = history + gradient ** 2
    alpha = initial_alpha / np.sqrt(history)
    
    return alpha, history
