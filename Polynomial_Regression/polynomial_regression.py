# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 10:49:43 2022

@author: oussama
Polynomial Regression: No Linear model
"""

#Data preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1.importing the dataset: ===> to a dataFrame : matrice [Line, columns]
dataset = pd.read_csv('Position_Salaries.csv')
#in model ML ===> matrix of feature consider all the time as matrex not a column: 1:2
#2.get the independent variables
X = dataset.iloc[:, 1:2].values
 
#3.get the dependent values: One vector => ecause just one variable
y = dataset.iloc[:, 2].values 
#We have a small data for observations ===> take all data for training no test set

#4.fitting Linear Regression to the dataset : include feature scaling
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#5. Fitting Polynomial Regression to the dataSet
from sklearn.preprocessing import PolynomialFeatures
#object poly_reg transform our matrex X to a new matrix feature that contain X1, X1 power 2, ....
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
#create a new regressor and fitt it
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#6.Visualising the linear Regression results
plt.scatter(X, y, color = 'red')
#plot the prediction:
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Linear Regression')
plt.xlabel('posiion Level')
plt.ylabel('Salary')
plt.show()


#7.Visualising the polynomial Regression results
#specify the increment level
X_grid = np.arange(min(X), max(X), 0.1)
#we nedd X_grid to be a matrix
X_grid =  X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
#plot the prediction:
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Polynomial Regression')
plt.xlabel('position Level')
plt.ylabel('Salary')
plt.show()

#predecting  a new result with Linear regression
lin_reg.predict(np.array([6.5]).reshape(1, 1))
"""
Scikit does not work with scalars (just one single value). It expects a shape (m×n) where m is the number of features and n is the number of observations, both are 1 in your case.
"""
#predecting  a new result with polynomial regression
lin_reg2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(1, 1)))







