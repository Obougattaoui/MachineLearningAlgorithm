#Data Preprocessing:
#all libraries that we need are selected by default/and we can install more libraries

#importing the dataset:
dataset = read.csv('Data.csv') #indexed start with 1 and not with 0
#we dont have make a distinction between matrix of features and dependent variables vector

#Taking care of missing DATA:
dataset$Age = ifelse(is.na(dataset$Age),
                        ave(dataset$Age, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Age)
dataset$Salary = ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, FUN = function(x) mean(x, na.rm = TRUE)),
                        dataset$Salary)

#Encoding Categorical data:(categorical to numeric category but you see variable as factor and shoes the label of categories)
dataset$Country = factor(dataset$Country, 
                         levels = c('France', 'Spain', 'Germany'),
                          labels = c(1, 2, 3))
dataset$Purchased = factor(dataset$Purchased, 
                           levels = c('No', 'Yes'),
                           labels = c(0, 1))
#Splitting data set(Training set and Test Set): caTools
#install.packages('caTools')
library(caTools) # ===> include this library
set.seed(123) #random_state ===> to get the same result
#SplitRatio pourcentage for the training set
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
#split return true(if the observations choose it for Training set) of false(if the observations choose it for test set)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Feature scaling: we should exclude categorical from the feature scaling
training_set[, 2:3] = scale(training_set[, 2:3])
test_set = scale(test_set[, 2:3])
