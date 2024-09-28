---
title: Gradient Boosting and LOESS
categories:
- General
excerpt: |
  Gradient Boosting Algorithim using the LOESS function and Comparison of scalars
feature_text: |
  ## Assignment 2 Part 1
  Creating a class for the LOESS function
feature_image: "https://picsum.photos/2560/600?image=733"
image: "https://picsum.photos/2560/600?image=733"
---
# Gradient Boosting

## Introduction
Gradient Boosting is an ensemble learning method, meaning it uses multiple models to improve its predictive performance. By training weak learner models
iteratively, it seeks to improve performance by correcting the residuals in the previous model. The final prediction is the sum of all the individual predictions.

Gradient Boosting can be prone to overfitting due to its number of its iterations on training set errors and emphasis on outliers, so it's important to consider ways to mitigate overfitting our model. This can be done by introducing regularization or finetuning hyperparameters [1,2]. 

In this particular implementation, we have a locally weighted regression method ([Lowess class, I created in my previous project! ](https://emxee333.github.io/data-440-capstone//general/2024/09/13/proj1loess/)) that allows a user-prescribed number of boosting steps. 

## Implementation
```python
class MyBoostedLowess():
  def __init__(self, n_boosting_steps=10, learning_rate=0.01):
    self.n_boosting_steps = n_boosting_steps
    self.lr=learning_rate
    self.models = []

  def is_fitted(self):
    return hasattr(self, 'xtrain_') and hasattr(self, 'ytrain_')

  def fit(self, x, y):
    self.xtrain_ = x
    self.ytrain_ = y

    prediction = np.zeros(len(y))

    for i in range(self.n_boosting_steps):
      residuals = y - prediction
      fitted_model, _ = self.__fit_one_step(x, residuals)
      # Update the prediction by adding the new model’s predictions, scaled by the learning rate
      prediction+= self.lr * fitted_model.predict(x).flatten()
      self.models.append(fitted_model)

  def __fit_one_step(self, x, y):
    fitted_model = MyLowess()
    fitted_model.fit(x,y)
    yhat = fitted_model.predict(x)
    residuals = y - yhat
    return fitted_model, residuals

  def predict(self, xnew):
    if self.is_fitted() == False:
      raise Exception('Model is not fitted yet')

    prediction = np.zeros((xnew.shape[0],1))

    for model in self.models:
      prediction +=self.lr * model.predict(xnew).reshape(-1,1)

    return prediction

  def get_params(self, deep=True):
    return {'n_boosting_steps': self.n_boosting_steps, 'learning_rate': self.lr}

  def set_params(self, **params):
    for param, value in params.items():
      setattr(self, param, value)
    return self

  def score(self, x, y):
    yhat = self.predict(x)
    return mse(y,yhat,squared=False)

```

## How does scaling the data affect our model?
Data scaling transforms the dataset to a common scale aross all variables. This can improve model performance, make comparison between variables.

StandardScaler follows the normal distribution, meaning the data is rescaled to unit variance and the mean = 0. The MinMaxScaler scales the data to [0,1] or [-1,1] if there are negative values in the dataset. The QuantileTransformer scales the data to have a Gaussian or uniform probability distribution.

To compare the differences between the scalers, I did a 10-fold cross-validation with each of the scalers:

```python
kf = KFold(n_splits=10,shuffle=True,random_state=1234)

boostedmodel = MyBoostedLowess()
scalers = [StandardScaler(), MinMaxScaler(), QuantileTransformer()]
rmse_dict= {'StandardScaler': None, 'MinMaxScaler': None, 'QuantileTransformer': None}
rmse_keys= list(mse_dict.keys())

for i in range(len(scalers)):
  scale = scalers[i]
  for idxtrain, idxtest in kf.split(x):
    xtrain = x[idxtrain]
    ytrain = y[idxtrain]
    ytest = y[idxtest]
    xtest = x[idxtest]
    xtrain = scale.fit_transform(xtrain)
    xtest = scale.transform(xtest)

    boostedmodel.fit(xtrain,ytrain)
    yhat_lw = boostedmodel.predict(xtest)
    rmse_dict[mse_keys[i]]=(root_mean_squared_error(ytest,yhat_lw))

  print('The Cross-validated Root Mean Squared Error for Locally Weighted Regression with the scaler '+ rmse_keys[i] + ' is: ' \ str(np.mean(mse_dict[mse_keys[i]])))
```

The resulted output is as follows:
The Cross-validated Root Mean Squared Error for Locally Weighted Regression with the scaler StandardScaler is: 26.710550810405948
The Cross-validated Root Mean Squared Error for Locally Weighted Regression with the scaler MinMaxScaler is: 24.072497870070347
The Cross-validated Root Mean Squared Error for Locally Weighted Regression with the scaler QuantileTransformer is: 23.13218421808465

## Hyperparameter Optimization and XGBoost Comparison

Using the “Concrete” data set, we want determine a choice of hyperparameters that yield lower MSEs for your method when compared to the eXtream Gradient Boosting library.

The XGBoost model [].

```python
from sklearn.model_selection import GridSearchCV

# Define the parameter grid to search
param_grid = {
    'n_boosting_steps': [5, 10, 20, 30],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Create a GridSearchCV object
grid_search = GridSearchCV(
    MyBoostedLowess(),
    param_grid,
    scoring='neg_mean_squared_error',  # Use negative MSE for maximization
    cv=10,  # Use 10-fold cross-validation
    n_jobs=-1  # Use all available cores
)

# Fit the grid search to the data
scale = StandardScaler()
xtrain = scale.fit_transform(xtrain)
xtest = scale.transform(xtest)
grid_search.fit(xtrain, ytrain)

# Print the best parameters and the best score
print("Best parameters: ", grid_search.best_params_)
print("Best score (negative MSE): ", grid_search.best_score_)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(xtest)
mse_best = mse(ytest, y_pred)
print("MSE on the test set with best model: ", mse_best)
```

```python
xgb_model = xgb.XGBRegressor()
xgb_model.fit(xtrain, ytrain)
y_pred_xgb = xgb_model.predict(xtest)
mse_xgb = mse(ytest, y_pred_xgb)
print("MSE on the test set with XGBoost: ", mse_xgb)

# Check if our best model performs better than XGBoost
if mse_best < mse_xgb:
  print("Our best model performs better than XGBoost.")
else:
  print("XGBoost performs better than our best model.")

```
### References
1. <https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502>
2. <https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/>


