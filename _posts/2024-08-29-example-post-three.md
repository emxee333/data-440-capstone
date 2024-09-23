---
title: Implementing a LOESS Function
categories:
- General
excerpt: |
  LOESS (Locally Estimated Scatterplot Smoothing) is a non-parametric regression technique used to fit a smooth curve through a set of data points.
feature_text: |
  ## Assingment 1
  Creating a class for the LOESS function
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---

Importing necessary libraries:

```python
# Libraries of functions need to be imported
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet,Lasso
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import cross_val_score as cvs
from scipy import linalg
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from scipy.spatial.distance import cdist

# the following line(s) are necessary if you want to make SKlearn compliant functions
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
```


Defining kernel functions

```python
# Gaussian Kernel
def Gaussian(x):
  return np.where(np.abs(x)>4,0,1/(np.sqrt(2*np.pi))*np.exp(-1/2*x**2))

# Tricubic Kernel
def Tricubic(x):
  return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)

# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,x,3/4*(1-np.abs(x)**2))

# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)
```

Now creating my class:

```python
class MyLowess():
  def __init__(self, kernel=Gaussian, tau=0.2, model=Ridge()):
    self.kernel = kernel
    self.tau = tau
    self.model=model

  def compute_weights(self, x, xnew):
    distances = cdist(x, xnew, metric='euclidean')
    if np.isscalar(xnew):
      w = self.kernel(distances/(2*self.tau))
    elif len(xnew.shape)==1:
        w = self.kernel(distances/(2*self.tau)).reshape(1,-1)
    else:
        w = self.kernel(distances/(2*self.tau))
    return w

  def fit(self, x, y):
    kernel = self.kernel
    tau = self.tau
    self.xtrain_ = x
    self.ytrain_ = y

  def predict(self, xnew):
    check_is_fitted(self)
    x = self.xtrain_
    y = self.ytrain_

    if np.ndim(xnew) == 1:
      weights = self.compute_weights(x,xnew)
      self.model.fit(np.diag(weights)@x.reshape(-1,1),np.diag(weights)@y.reshape(-1,1))
      yhat=self.model.predict(xnew)

    else:
      n=len(xnew)
      yhat = []
      weights = self.compute_weights(x,xnew)
      for i in range(n):
        self.model_ = self.model.fit(np.diag(weights[:,i])@x,np.diag(weights[:,i])@y)
        yhat.append(self.model.predict(xnew[i].reshape(1,-1)))

    return np.array(yhat)
```




