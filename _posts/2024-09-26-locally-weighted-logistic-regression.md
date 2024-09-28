---
title: Locally Weighted Logistic Regression
categories:
- General
excerpt: |
  LOESS (Locally Estimated Scatterplot Smoothing) is a non-parametric regression technique used to fit a smooth curve through a set of data points.
feature_text: |
  ## Assignment 2 Part 1
  Creating a class for Locally Weighted Logistic Regression
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---
### Locally Weighted Logistic Regression

#### Introduction
Logistic regression is used to predict the probability of each data entry belonging to a certain class. The logistic regression takes the linear regressio formula and transforms it using the sigmoid function. 

$${\displaystyle p(x)={\frac {1}{1+e^{-(\beta _{0}+\beta _{1}x)}}}}$$

In this case, we are using the softmax function, which can be used for multi-classification problems.


#### Implementation

```python

class LocallyWeightedLogisticRegression:
    def __init__(self, lr=0.01, max_iter=100000,  reg=1e-6, tau=0.1,threshold=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tau = tau  # bandwidth parameter for locality
        self.reg= reg
        self.threshold=threshold

    def __sigmoid(self, z):
      return 1 / (1 + np.exp(-z))

    def __softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __weight_matrix(self, X, x_query):
        # Calculate the distance from the query point to each training point and use Gaussian weighting
        m = X.shape[0]
        W = np.exp(-np.sum((X - x_query) ** 2, axis=1) / (2 * self.tau ** 2))
        return np.diag(W)

    def fit(self, X, y):
      if len(np.unique(y)) ==2:
        self.theta = np.zeros(X.shape[1])
        logit_func=self.__sigmoid
        num_classes = 2

      else:
        self.theta = np.zeros((X.shape[1], len(np.unique(y))))  # Multiclass (K classes)
        logit_func=self.__softmax
        num_classes=len(np.unique(y))

      for i in range(self.max_iter):
        total_gradient = np.zeros_like(self.theta)
        for j in range(X.shape[0]):
          x_query = X[j]

          if num_classes > 2:
              y_one_hot = np.eye(num_classes)[y]  # Shape: (m, num_classes)
          else:
              y_one_hot = y.reshape(-1, 1)  #one hot encoding of target var

          # Calculate the weight matrix based on the locality of the query point
          W = self.__weight_matrix(X, x_query)
          preds=logit_func(np.dot(X, self.theta))

          # Compute the gradient with locally weighted data
          gradient = (np.dot(X.T, np.dot(W, (preds - y_one_hot))) / y.shape[0]) + self.reg * self.theta

          # Accumulate the gradient for all training examples
          total_gradient += gradient

        # Update the weights using the gradient
        self.theta -= self.lr * gradient / X.shape[0]

        if np.linalg.norm(gradient) < self.threshold:
          break


    def predict_prob(self, X):
      z = np.dot(X, self.theta)
      if self.theta.ndim == 1:
        return self.__sigmoid(z)
      else:
        return self.__softmax(z)

    def predict(self, X):
        if self.theta.ndim == 1:  # Binary classification
            return (self.predict_prob(X) >= 0.5).astype(int)
        else:  # Multiclass classification
            return np.argmax(self.predict_prob(X), axis=1)

```

#### Evaluation
Using the Iris dataset:


```python
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score as cvs
from scipy import linalg
from sklearn.model_selection import KFold
import numpy as np

x = iris.loc[:,'SepalLengthCm':'PetalWidthCm'].values
y = iris['Species'].values


#target values are categorical, so we use LabelEncoder() to encode them into integers

le = LabelEncoder() 
y = le.fit_transform(y)

xtrain, xtest, ytrain, ytest = tts(x,y,test_size=0.2,random_state=1234)
```

Using a 10-fold cross validation to evaluate the model using mean squared error:

```python

kf = KFold(n_splits=10,shuffle=True,random_state=1234)

lwlr_model = LocallyWeightedLogisticRegression()
lwlr_mse=[]
scaler=StandardScaler()
for idxtrain, idxtest in kf.split(x):
  xtrain = x[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = x[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)


  lwlr_model.fit(xtrain,ytrain)
  yhat_pred_prob = lwlr_model.predict_prob(xtest)
  yhat_pred=lwlr_model.predict(xtest)
  lwlr_mse.append((mse(ytest,yhat_lw)))

print('The Cross-validated Root Mean Squared Error for Locally Weighted Regression is : '+str(np.mean(lwlr_mse)))
```


### References
1. <https://www.cs.cmu.edu/~kdeng/thesis/logistic.pdf>
2. <https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/>


