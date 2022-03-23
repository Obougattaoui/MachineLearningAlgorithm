#Regression Template : for No linear regression models

#Importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]


#we will not Splitting the dataset ==> we just have a small dataset


#Fitting xxxxxxx Regression to the dataset 
#=====> Create our regressor


#Predecting a new result with xxxxxx Regression:
y_pred = predict(regressor, newdata = data.frame(Level = 6.5))


#Visualizing the xxxxxxx regression results
#specify step = 0.1
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.1)

ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))),
            colour = 'blue') + 
  ggtitle('xxxxxxx Regression: Level VS Salary') +
  xlab('Level') +
  ylab('Salary')

