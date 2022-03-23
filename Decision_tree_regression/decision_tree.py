# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:56:33 2022

@author: oussama

Decision Tree Regression

"""

#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1.importing the dataset: ===> to a dataFrame : matrice [Line, columns]
dataset = pd.read_csv('Position_Salaries.csv')
#2.get the independent variables
X = dataset.iloc[:, 1:2].values
 
#3.get the dependent values: One vector => ecause just one variable
y = dataset.iloc[:, 2].values 

"""
#4.splitting the dataset into the training set and test set:
from sklearn.model_selection import train_test_split
#8 ==> int the train set and 2=> in the test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
"""

#5.Fitting the decision tree Regression model to the dataSet
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X, y)



#6.Predecting  a new result with polynomial regression
y_pred = regressor.predict(np.array([6.5]).reshape(1, 1))

#8.Visualising the Regression results(for highere resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Decision Tree  Regression Model')
plt.xlabel('position Level')
plt.ylabel('Salary')
plt.show()






