import numpy as np

def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    output = []
    # Write code here
    for i in x:
        if i>0:
            output.append(i)
        elif i<=0:
            output.append(alpha*(np.exp(i)-1))

    return output