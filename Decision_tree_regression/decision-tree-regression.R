#Decision Tree Regression:

#Importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]


#we will not Splitting the dataset ==> we just have a small dataset


#Fitting Decision Tree Regression to the dataset 
install.packages('rpart')
library(rpart)
regressor = rpart(formula = Salary ~ .,
                  data = dataset,
                  control = rpart.control(minsplit = 1))


#Predecting a new result with Decision Tree Regression:
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))





#Visualizing the Decision Tree regression results
#specify step = 0.1
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') + 
  ggtitle('Decision Tree Regression: Level VS Salary') +
  xlab('Level') +
  ylab('Salary')

