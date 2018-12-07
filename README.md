# Variable_Neural_Net Without MNIST

This code takes the no of layers as variable. It also takes the no of input layer nodes and output layer nodes as input. they must match properly with the given data set. here is a sample data set given

Change the learning rate at the 3rd line of code as per requirement

# For using with MNIST

### Requirements - glob , numpy and open-cv

Download the data set from https://pjreddie.com/media/files/mnist_test.tar.gz
Extract the file and keep it in the same folder as that of the code . The filenames here actually tell their label.
Here you can use as many data-points as you want as this does not load the entire dataset during backpropagation
Change the no of test and train images at the 6th and 7th line of codeand the learning rate at the 5th line

# For using the general one

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
