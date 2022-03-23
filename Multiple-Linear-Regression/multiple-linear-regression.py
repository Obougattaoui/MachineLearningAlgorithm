# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 14:57:49 2022

@author: oussama
"""

#               Multiple Linear Regression
#Data preprocessing
#importing the libraries :
import numpy as np #Mathematical tools
import matplotlib.pyplot as plt # Matlab library to plot 
import pandas as pd  #to manipulate the data set

#1.importing the dataset: ===> to a dataFrame : matrice [Line, columns]
dataset = pd.read_csv('50_Startups.csv')

#2.get the independent variables
X = dataset.iloc[:, 0:-1].values
 
#3.get the dependent values: One vector => ecause just one variable
y = dataset.iloc[:, 4].values 

#5.categorical data: Encode target labels with value between 0 and n_classes-1.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#In this technique, we each of the categorical parameters, it will prepare separate columns for both Male and Female label. SO, whenever there is Male in Gender, it will 1 in Male column and 0 in Female column and vice-versa.
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough')
#A ColumnTransformer takes in a list, which contains tuples of the transformations we wish to perform on the different columns. Each tuple expects 3 comma-separated values: first, the name of the transformer, which can be practically anything (passed as a string), second is the estimator object, and the final one being the columns upon which we wish to perform that operation.
X = onehotencoder.fit_transform(X)

#Avoiding the Dummy variable Trap:
X = X[:, 1:]#remove the first variable columln

#splitting the dataset into the training set and test set:
from sklearn.model_selection import train_test_split
#8 ==> int the train set and 2=> in the test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#As the docs mention, random_state is for the initialization of the random number generator used in train_test_split (similarly for other methods, as well). As there are many different ways to actually split a dataset, this is to ensure that you can use the method several times with the same dataset (e.g. in a series of experiments) and always get the same result (i.e. the exact same train and test sets here), i.e for reproducibility reasons. Its exact value is not important and is not something you have to worry about.


#Fitting multiple Linear Regression to the training set :
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()#create an object of LinearRegression class
#fit the object to our training set :
regressor.fit(X_train, y_train)

#Predecting the test set results :
y_pred = regressor.predict(X_test)

#building the optimat model using backward elimination
import statsmodels.api as sm
#append b0 to our matrix
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
#dftype ????? #ones will return a matrix of only one value inside
#matrix optimal : contains independent var have high impact on the profit :
X_opt  = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float) 
#fit the full model with all possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
"""P-value is the probability for the Null hypothesis to be true
    Null hypothesis treats everything same everything equal
    effect of do things are the same
    0<= p-value <= 1
    p = 0.1
        10 of 100 null hypothesis will be true(if i repeat the experiment 100 times)
"""
"""In inferential statistics, the null hypothesis (often denoted H0)[1] is that there is no difference between two possibilities. The null hypothesis is that the observed difference is due to chance alone. Using statistical tests it is possible to calculate the likelihood that the null hypothesis is true.

"""
regressor_OLS.summary()
