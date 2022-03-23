#set the working directory
#Importing the dataset:
dataset = read.csv('Data.csv')
#dataset = dataset[, 2:3]
#splitting the dataset into the training set and test set :
#install.packages('caTools')
#library(caTools)
set.seed(123)

#spliting method: split
#sample.split==> only set dependent vector :
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
#Result : True : training set and False: test set
trainig_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# #Feature scaling :
# trainig_set[, 2:3] = scale(trainig_set[, 2:3])
# test_set[, 2:3] = scale(test_set[, 2:3])
