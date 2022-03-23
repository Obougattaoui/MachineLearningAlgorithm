# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 17:34:15 2022

@author: oussama
Multiple Linear regression version 2
"""

"""
First step preparing the dataset
"""
#Data preprocessing
#importing the libraries :
import numpy as np #Mathematical tools
import matplotlib.pyplot as plt # Matlab library to plot 
import pandas as pd  #to manipulate the data set

#1.importing the dataset: ===> to a dataFrame : matrice [Line, columns]
dataset = pd.read_csv('50_Startups.csv')

#2.get the matrix of the independent variables
X = dataset.iloc[:, 0:-1].values
 
#3.get the dependent values: One vector => because just one variable
y = dataset.iloc[:, 4].values 


#4.Create the Dummy variables (categorical variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
#change the text into numbers
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
#In this technique, we each of the categorical parameters, it will prepare separate columns for both Male and Female label. SO, whenever there is Male in Gender, it will 1 in Male column and 0 in Female column and vice-versa.
onehotencoder = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],
    remainder='passthrough')
#A ColumnTransformer takes in a list, which contains tuples of the transformations we wish to perform on the different columns. Each tuple expects 3 comma-separated values: first, the name of the transformer, which can be practically anything (passed as a string), second is the estimator object, and the final one being the columns upon which we wish to perform that operation.
X = onehotencoder.fit_transform(X)


#5.Avoiding the dummy variable Trap
X = X[:, 1:] #☺removing the first column from X
      

#6.splitting the dataset into the training set and test set:
from sklearn.model_selection import train_test_split
#8 ==> int the train set and 2=> in the test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0) 
#is not important and is not something you have to worry about.

#7.♠fittting multiple Linear regression to the training ser
from sklearn.linear_model import LinearRegression
regressor = LinearRegression() #regressor is an object of LinearRegression Class
#fit the object
regressor.fit(X_train, y_train)

#8.Predecting the test results
y_pred = regressor.predict(X_test)

#9. add the X0 for the constant b0
#add a column of 1 ===> associate to b0 constant
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
# we should convert this column to integer type ==> Model ==> just integers
#axis to specify if we want to add a column or a line
#to add to begining of the matrix : add to the column to the x


#9.building the optimal model using backword elimination
import statsmodels.api as sm
#Create a new Optimal matrix : 
X_opt  = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float) 
#step2: create a new regressor(we should fit our regressor(Bachward elimination) for X_opt an y)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#predictors = IV
#p_value for each independent value and compare it to the SV 
regressor_OLS.summary()#Step3 : select the predictor X2 for th column 2 because he has the highest p-value
#step 4: remove X2
X_opt  = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float) 
#step5: fit the model without X2
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#step4 : remove X1 fro column 1
X_opt  = np.array(X[:, [0, 3, 4, 5]], dtype=float) 
#step5: fit the model without X1
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#step4 : remove X2 for column 4
X_opt  = np.array(X[:, [0, 3, 5]], dtype=float) 
#step5: fit the model without X1
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#step4 : remove X2 for column 5
X_opt  = np.array(X[:, [0, 3]], dtype=float) 
#step5: fit the model without X1
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()





