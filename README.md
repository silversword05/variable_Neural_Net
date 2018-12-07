# Variable_Neural_Net

## For using with MNIST

### Requirements - glob , numpy and open-cv

Download the data set from https://pjreddie.com/media/files/mnist_test.tar.gz
Extract the file and keep it in the same folder as that of the code . The filenames here actually tell their label.
Here you can use as many data-points as you want as this does not load the entire dataset during backpropagation
Change the no of test and train images at the 6th and 7th line of codeand the learning rate at the 5th line

## For using the general one

The instruction file requitements are as follows
- first line should be learning rate
- second line should contain no of iterations
- third line should contain no of nodes in input layer
- fourth line should contain no of nodes in output layer
- fifth line should contain normalization factor
- sixth line should contain no of hidden layer
- seventh line should contain no of nodes in hidden layers respectively

Sample instructions are given
Sample data is also given which is created from MNIST database

In the data file, each data-point occur in each line. The values are separated by space and the last value is the label.

The weight matrix file contains each row in one line and in between two matrices, they are separated by two lines.
### The weight matrix is that matrix for which minimum error has occured on the training set which is also displayed at the last as minimum error
### During calculation of cost function , the one with the assigned label and ones with values greater than the assigned label are kept. All others are intentionally ignored by making them zero and then applying Back-Prop Algorithm so that they do not effect the weight gradients. This is done as during prediction , the one with the highest value among the output nodes is taken.
### Learning rate during entire training is kept constant.
