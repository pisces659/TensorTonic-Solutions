def gradient_descent_quadratic(a,b,c,x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    
    # Write code here
    #ActualY = x0**2-4*x0+3

    x = x0
    for i in range(steps):
        GradY = 2*a*x+b
        x = x - GradY*lr
    
    return x



