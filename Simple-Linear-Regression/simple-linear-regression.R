#simple Linear Regression:
#Data Preprocessing:
#all libraries that we need are selected by default/and we can install more libraries

#importing the dataset:
dataset = read.csv('Salary_Data.csv')

#Splitting data set(Training set and Test Set): caTools
#install.packages('caTools')
library(caTools) # ===> include this library
set.seed(123) #random_state ===> to get the same result
#SplitRatio pourcentage for the training set
split = sample.split(dataset$Salary, SplitRatio = 2/3)
#split return true(if the observations choose it for Training set) of false(if the observations choose it for test set)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#fitting simple linear Regression to the training set:
# Salary ~ YearsExperience ===> a simple linear formula/ data ==> to train our simple linear model
regressor = lm(formula =Salary ~ YearsExperience,
              data = training_set)

#Predecting the test set results:
y_pred = predict(regressor, newdata = test_set)

#Visualising the Training set results:
install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary), 
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Training Set: Salary VS YearsOfExperience') + 
  xlab('Years of experience') + 
  ylab('Salary')
  
ggplot() +
  geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary), 
             colour = 'red') +
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, newdata = training_set)),
            colour = 'blue') +
  ggtitle('Test set: Salary VS YearsOfExperience') + 
  xlab('Years of experience') + 
  ylab('Salary')




