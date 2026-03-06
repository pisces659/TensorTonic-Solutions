import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.asarray(x)
    Output=[]

    RELU = np.maximum(0,x).tolist()
    return np.asarray(RELU)
    