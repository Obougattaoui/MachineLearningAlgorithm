#Polynomial Regression: No Linear Regression models(No linear intership between Independent variables)


#Importing dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[, 2:3]


#we will not Splitting the dataset ==> we just have a small dataset


#Fitting Linear Regression to the dataset
lin_reg = lm(formula = Salary ~ .,
             data = dataset)

#Fitting Polynomial Regression to the dataset /Level2 => Level Squared
dataset$Level2 = dataset$Level^2
dataset$Level3 = dataset$Level^3
dataset$Level4 = dataset$Level^4
#dataset$Level5 = dataset$Level^5
#dataset$Level6 = dataset$Level^6
#dataset$Level7 = dataset$Level^7
#dataset$Level8 = dataset$Level^8
poly_reg = lm(formula = Salary ~ .,
              data = dataset)


#Visualizing the Linear regression results
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(lin_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Linear Regression: Level VS Salary') +
  xlab('Level') +
  ylab('Salary')


#Visualizing the polynomial regression results
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(poly_reg, newdata = dataset)),
            colour = 'blue') + 
  ggtitle('Polynomial Regression: Level VS Salary') +
  xlab('Level') +
  ylab('Salary')


#Predecting a new result with Linear Regression:
y_pred = predict(lin_reg, newdata = data.frame(Level = 6.5))


#Predecting a new result with Polynomial Regression:
y_pred = predict(poly_reg, newdata = data.frame(Level = 6.5,
                                                Level2 = 6.5^2,
                                                Level3 = 6.5^3,
                                                Level4 = 6.5^4))

