#Simple Linear Regression :
#Data preprocessing
#importing the libraries :
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

#importing the dataset:
dataset = pd.read_csv('Salary_Data.csv')
#matrice [Line, columns], : ==> ALL
#:-1 ==> all the columns except the last one
#create the independent variable :
X = dataset.iloc[:, :-1].values
#Create the dependent Variable:
y = dataset.iloc[:, 1].values

#splitting the dataset into the training set and test set:
from sklearn.model_selection import train_test_split
#8 ==> int the train set and 2=> in the test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 1/3, random_state = 0)


#Fitting simple Linear Regression to the training set :
from sklearn.linear_model import LinearRegression
#regressor is machine learning model :
regressor = LinearRegression()
regressor.fit(X_train, y_train)#make the model learn 
#before execution our model already learn

#Predecting the test set result :
#put salaries in a vector of predecting y_pred
y_pred = regressor.predict(X_test)

#visulising the training set results :
plt.scatter(X_train, y_train, color = 'red')
#plot the regression line :
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#predect new observation Test:
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary Vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()    
#Feature Scaling :
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
#test ==> we dont need to fit
X_test = sc_X.transform(X_test)"""