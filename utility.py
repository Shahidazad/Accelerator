import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Create a function to map the features
def mapFeature(X1, X2, degree):
    """
    Create a new feature for each polynomial degree of X1 and X2

    Parameters:
        X1: The first feature
        X2: The second feature
        degree: The polynomial degree

    Returns:
        The new features
    """
    res = np.ones(X1.shape[0])
    for i in range(1,degree + 1):
        for j in range(0,i + 1):
            res = np.column_stack((res, (X1 ** (i-j)) * (X2 ** j)))
    return res



# Sigmoid function
def sigmoid(z):
    """
    Computes the sigmoid function.

    Parameters:
        z: The input to the sigmoid function.

    Returns:
        The output of the sigmoid function.
    """
    return 1 / (1 + np.exp(-z))



y_pred=0
def costFunc(theta, X, y):
    """
    Computes the cost function for logistic regression.

    Parameters:
    theta: The parameters of the logistic regression model.
    X: The training data.
    y: The labels for the training data.

    Returns:
    The cost of the logistic regression model.
    """

    # Number of training examples

    m = y.shape[0]

    # Calculate the predictions of the logistic regression model

    z = X.dot(theta)
    h = sigmoid(z)

    # Save the predictions for later use

    global y_pred
    y_pred = np.where(h >= 0.5, 1, 0)

    # Calculate the cost

    term1 = y * np.log(h)
    term2 = (1 - y) * np.log(1 - h)
    J = -np.sum(term1 + term2, axis=0) / m

    return J

y_pred=y_pred

# Plotting Decision Boundary for Binary Classification
def plotDecisionBoundary(theta, degree, axes):
    # Generate grid points for plotting the decision boundary
    u = np.linspace(-1.5, 3, 50)
    v = np.linspace(-1.5, 3, 50)
    U, V = np.meshgrid(u, v)

    # Convert U, V to vectors for calculating additional features using vectorized implementation
    U = np.ravel(U)
    V = np.ravel(V)
    Z = np.zeros((len(u) * len(v)))

    # Generate polynomial features for the grid points
    X_poly = mapFeature(U, V, degree)

    # Calculate the predicted values for the grid points
    Z = X_poly.dot(theta)

    # Reshape U, V, Z back to matrix form for contour plotting
    U = U.reshape((len(u), len(v)))
    V = V.reshape((len(u), len(v)))
    Z = Z.reshape((len(u), len(v)))

    # Plot the decision boundary as a contour plot
    cs = axes.contour(U, V, Z, levels=[0], cmap="Greys_r")
    
    # Add legend to the plot
    axes.legend(labels=['diabetic', 'normal', 'Decision Boundary'])
    plt.show()
    return cs


def subplot(poly_df,pos,neg,degree):
    """
    Generate a subplot with scatter plots representing positive and negative classes.

    Args:
        poly_df (DataFrame): A pandas DataFrame containing the data points.
        pos (array-like): Indices of positive class data points.
        neg (array-like): Indices of negative class data points.
        degree (int): Degree of the decision boundary.

    Returns:
        fig (Figure): The generated figure object.
        axes (Axes): The axes object representing the subplot.
        text (str): Report text based on the degree of the decision boundary.

    """

    fig, axes = plt.subplots(figsize=(12, 7))
    axes.set_xlabel('Feature 1')
    axes.set_ylabel('Feature 2')

    # Scatter plot for positive and negative class
    axes.scatter(poly_df.loc[pos, 0], poly_df.loc[pos, 1], color='r', marker='x', label='diabetic')
    axes.scatter(poly_df.loc[neg, 0], poly_df.loc[neg, 1], color='g', marker='o', label='normal')
    # Add legend
    axes.legend(title='Legend', loc='best')
    # Set title and create report text based on degree
    if degree == 1:
        axes.set_title('Linear Decision Boundary for logistic regression')
        text = '\033[1m\033[34mLinear Decision Boundary Report\033[0m'
    elif degree == 2:
        axes.set_title('Quadratic Decision Boundary for logistic regression')
        text = '\033[1m\033[34mQuadratic Decision Boundary Report\033[0m'
    elif degree == 3:
        axes.set_title('Cubic Decision Boundary for logistic regression')
        text = '\033[1m\033[34mCubic Decision Boundary Report\033[0m'
    else:
        axes.set_title(f'{degree}th Degree Decision Boundary for logistic regression')
        text = f'\033[1m\033[34m{degree}th Degree Decision Boundary Report\033[0m'

    plt.show()
        
    return fig,axes,text


# Plotting Decision Boundary for Binary Classification
# The function plots the decision boundary for a given SVM model using the provided test data.

def plot_decision_boundary(model, X, y,df_test,pos,neg):
    """
    Plot the decision boundary for a given SVM model and the corresponding data.
    The function generates a mesh grid based on the feature ranges of the data and uses the trained model to predict the class labels for all points in the mesh grid. It then plots the decision boundary along with the data points, highlighting positive class instances with a red 'x' marker and negative class instances with a green 'o' marker. 

    Parameters:
    - model: The trained SVM model.
    - X: The feature matrix of the data.
    - y: The target labels of the data.
    - df_test: DataFrame containing the test data.
    - pos: Boolean mask for positive class instances.
    - neg: Boolean mask for negative class instances.

    Returns:
    None (Displays the plot.)

    """
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the class labels for all points in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

    # Reshape the predicted labels to match the mesh grid shape
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary and the data points
    fig, axes = plt.subplots(figsize=(12,7));
    # axes.figure(figsize=(12,7))
    axes.set_xlabel('Feature 1')
    axes.set_ylabel('Feature 2')
    axes.set_title('Poly Decision Boundary for SVM  ')
    text='\033[1m\033[SVM Decision Boundary Report\033[0m'
    plt.contour(xx, yy, Z, alpha=0.8,colors='black')
    axes.scatter(df_test.loc[pos, 0], df_test.loc[pos, 1], color = 'r', marker='x', label='diabetic')
    axes.scatter(df_test.loc[neg, 0], df_test.loc[neg, 1], color = 'g', marker='o', label='normal')
    axes.legend(title='Legend', loc = 'best' )
    plt.show()