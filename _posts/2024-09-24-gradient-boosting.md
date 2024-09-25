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
### Gradient Boosting
Gradient Boosting is an ensemble learning method, meaning it uses multiple models to improve its predictive performance. By training weak learner models
iteratively, it seeks to improve performance by correcting the residuals in the previous model. The final prediction is the sum of all the individual predictions.

Gradient Boosting can be prone to overfitting, so it's important to consider [something lmao].

In this particular implementation,  locally weighted regression method (Lowess class), and that allows a user-prescribed number of boosting steps. 


#### Show applications with real data for regression, 10-fold cross-validations and compare the effect of different scalers, such as the “StandardScaler”, “MinMaxScaler”, and the “QuantileScaler”. 

StandardScaler uses the z-score

MinMaxScaler scales the data to [0,1]

QuantileScaler

#### In the case of the “Concrete” data set, determine a choice of hyperparameters that yield lower MSEs for your method when compared to the eXtream Gradient Boosting library.

- Elbow method
- GridSearchCV



### References
1. <https://towardsdatascience.com/all-you-need-to-know-about-gradient-boosting-algorithm-part-1-regression-2520a34a502>
2. <https://machinelearningmastery.com/gentle-introduction-gradient-boosting-algorithm-machine-learning/>


