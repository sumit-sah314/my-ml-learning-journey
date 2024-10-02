import numpy as np
import matplotlib.pyplot as plt

# Hypothesis function for linear regression
def hypothesis(X, theta):
    return np.dot(X, theta)

# Compute the cost function
def compute_cost(X, y, theta):
    m = len(y)
    return (1/(2*m)) * np.sum(np.square(hypothesis(X, theta) - y))

# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        # Compute gradient
        theta = theta - (learning_rate/m) * np.dot(X.T, (hypothesis(X, theta) - y))
        
        # Compute and save the cost
        cost_history[i] = compute_cost(X, y, theta)
    
    return theta, cost_history

# Example data
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  # Adding a column of 1s for bias (intercept term)
y = np.array([1, 2, 3, 4, 5])  # Target values
theta = np.zeros(2)  # Initialize parameters (slope and intercept)

# Gradient Descent settings
learning_rate = 0.01
iterations = 1000

# Perform gradient descent
theta, cost_history = gradient_descent(X, y, theta, learning_rate, iterations)

# Print the final parameters (theta)
print(f"Optimal parameters: {theta}")

# Plot the cost function vs. iterations
plt.plot(range(iterations), cost_history, color='blue')
plt.title("Cost Function over Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

# Plot the regression line
plt.scatter(X[:, 1], y, color='red', label="Data points")
plt.plot(X[:, 1], hypothesis(X, theta), color='blue', label="Regression line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
