import numpy as np
import math

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    
    Xarray = np.asarray(X)
    Yarray = np.asarray(y)

    N, D = Xarray.shape
    
    #Initialize weigths
    w = np.zeros(D) 
    b = 0

    
    for step in range(steps):
        cal = Xarray@w+b     
        pred = _sigmoid(cal)

        loss =[]
        for j in range(len(Xarray)):
           loss.append(y[j]*math.log(pred[j]) +  (1-y[j])*math.log(1-pred[j]))
        np.mean(loss)
    
        gradw = Xarray.T@(pred-Yarray)*(1/N)
        gradb = np.mean(pred-Yarray)
        
        w = w - lr*gradw
        b = b - lr*gradb
    
    return (w, b)















