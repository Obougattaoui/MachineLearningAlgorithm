#Data preprocessing
#importing the libraries :
import numpy as np #Mathematical tools
import matplotlib.pyplot as plt # Matlab library to plot 
import pandas as pd  #to manipulate the data set

#1.importing the dataset: ===> to a dataFrame : matrice [Line, columns]
dataset = pd.read_csv('Data.csv')

#2.get the independent variables
X = dataset.iloc[:, 0:-1].values
 
#3.get the dependent values: One vector => ecause just one variable
y = dataset.iloc[:, 3].values 

#splitting the dataset into the training set and test set:
from sklearn.model_selection import train_test_split
#8 ==> int the train set and 2=> in the test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
#As the docs mention, random_state is for the initialization of the random number generator 
#used in train_test_split (similarly for other methods, as well). As there are many different 
#ways to actually split a dataset, this is to ensure that you can use the method several times
# with the same dataset (e.g. in a series of experiments) and always get the same result 
#(i.e. the exact same train and test sets here), i.e for reproducibility reasons. Its exact value 
#is not important and is not something you have to worry about.

#Feature Scaling :
#StandardScaler follows Standard Normal Distribution (SND). Therefore, it makes mean = 0 and scales the data to unit variance. 
#MinMaxScaler scales all the data features in the range [0, 1] or else in the range [-1, 1] if there are negative values in the dataset. This scaling compresses all the inliers in the narrow range [0, 0.005]. 
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)"""
#test ==> we dont need to fit
#X_test = sc_X.transform(X_test)