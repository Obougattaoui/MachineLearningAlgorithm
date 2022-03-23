#Multiple Linear Regression:

#which of the independent variables has the highest effects to the dependent variable
#import dataset
dataset = read.csv('50_Startups.csv')

#State ===> is a categorical variable

#categorical data:
dataset$State = factor(dataset$State,
                       levels = c('New York', 'California', 'Florida'),
                       labels = c(1, 2, 3))


#Training set and test set:
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE) 
test_set = subset(dataset, split == FALSE) 

#Fitting multiple Linear regression to the training set
regressor = lm(formula = Profit ~ .,
               data = training_set)
# . ===> All independent variables

#summary(regression) ===> to show effects ...


#predecting the Test set results:
y_pred = predict(regressor,newdata = test_set)


#Building the optimal model using Backword Elimination:
#R&D ===> R.D(in R replace space by .)
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
               data = dataset)  
#we select dataset just to have complete information which IV are staticly significant


summary(regressor)


#removes State: State2 ===> 0.990 and State3 ===> 0.943
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend ,
               data = dataset)
summary(regressor)


#removes Administration ==> 0.602
regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend ,
               data = dataset)
summary(regressor)
 

#removes Marketing.Spend ==> 0.06
regressor = lm(formula = Profit ~ R.D.Spend ,
               data = dataset)
summary(regressor)

