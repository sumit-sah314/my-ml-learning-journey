import numpy as np

def compute_cost(X_b, y, theta):
    """
    Compute the cost function for linear regression.
    
    Parameters:
        X_b (numpy.ndarray): Feature matrix with a bias term (m x n+1).
        y (numpy.ndarray): Target values (m x 1).
        theta (numpy.ndarray): Parameters (n+1 x 1).
    
    Returns:
        float: The computed cost.
    """
    m = len(y)
    predictions = X_b.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

def gradient_descent(X_b, y, theta, learning_rate, n_iterations):
    """
    Perform gradient descent to optimize parameters for linear regression.
    
    Parameters:
        X_b (numpy.ndarray): Feature matrix with a bias term (m x n+1).
        y (numpy.ndarray): Target values (m x 1).
        theta (numpy.ndarray): Initial parameters (n+1 x 1).
        learning_rate (float): Learning rate for parameter updates.
        n_iterations (int): Number of iterations for gradient descent.
    
    Returns:
        numpy.ndarray: The optimized parameters.
        list: History of the cost function values.
    """
    m = len(y)
    cost_history = []
    
    for iteration in range(n_iterations):
        gradients = (1 / m) * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients
        cost_history.append(compute_cost(X_b, y, theta))
    
    return theta, cost_history

