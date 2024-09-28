---
title: Locally Weighted Logistic Regression
categories:
- General
excerpt: |
  LOESS (Locally Estimated Scatterplot Smoothing) is a non-parametric regression technique used to fit a smooth curve through a set of data points.
feature_text: |
  ## Assignment 2 Part 1
  Creating a class for the LOESS function
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---
### Locally Weighted Logistic Regression
Logistic regression is similar to a linear regression but noooooo something about two classes and a sigmoid 

```python
import numpy as np

class LocallyWeightedLogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False, tau=0.1):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.tau = tau  # bandwidth parameter for locality

    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def __softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # Stability trick
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __loss(self, h, y):
        return -np.mean(y * np.log(h + 1e-8))  # Add a small constant for numerical stability

    def __weight_matrix(self, X, x_query):
        # Calculate the distance from the query point to each training point and use Gaussian weighting
        m = X.shape[0]
        W = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * self.tau ** 2))
        return np.diag(W)

    def fit(self, X, y, x_query):
        if self.fit_intercept:
            X = self.__add_intercept(X)

        # Initialize weights (theta) for each class
        self.theta = np.zeros((X.shape[1], y.shape[1]))  # Multiclass (K classes)

        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__softmax(z)

            # Calculate the weight matrix based on the locality of the query point
            W = self.__weight_matrix(X, x_query)

            # Compute the gradient with locally weighted data
            gradient = np.dot(X.T, np.dot(W, (h - y))) / y.shape[0]
            self.theta -= self.lr * gradient

            if self.verbose and i % 10000 == 0:
                loss = self.__loss(h, y)
                print(f'Iteration {i}, Loss: {loss}')

    def predict_prob(self, X):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        z = np.dot(X, self.theta)
        return self.__softmax(z)

    def predict(self, X):
        return np.argmax(self.predict_prob(X), axis=1)

```
### References
1. <https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502>
2. <https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/>


