#Random Forest Regression:

#Importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]


#we will not Splitting the dataset ==> we just have a small dataset


#Fitting Random Forest Regression to the dataset 
install.packages('randomForest')
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],
                         y = dataset$Salary,
                         ntree = 500)


#Predecting a new result with Random Forest Regression:
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))


#Visualizing the Random Forest regression results
#specify step = 0.1
library(ggplot2)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') + 
  ggtitle('Random Forest Regression: Level VS Salary') +
  xlab('Level') +
  ylab('Salary')

