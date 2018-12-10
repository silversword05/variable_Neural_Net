# Variable_Neural_Net

### Requirements 
- glob , numpy and open-cv for the data file creation from the MNIST database
- numpy for training and getting the optimised weight matrices
- numpy for testing and doing predictions
- seaborn , pandas , matplotlib for creating confusion matrix

Also use 
```
sudo apt-get install python3-tk
```
Download the data set from https://pjreddie.com/media/files/mnist_test.tar.gz  
Extract the file and keep it in the same folder as that of the code. The filenames here actually tell their label.   
Here you can use as many data-points as you want as this does not load the entire dataset during backpropagation

## Instruction File and other formalities

The instruction file requitements are as follows
- first line should be learning rate
- second line should contain no of iterations
- third line should contain no of nodes in input layer
- fourth line should contain no of nodes in output layer
- fifth line should contain normalization factor
- sixth line should contain no of hidden layer
- seventh line should contain no of nodes in hidden layers respectively
- eight line 0 or 1 filename.txt -- **0** when weights are initialized randomly and **1 filename.txt** when initial weights are read from a file and the file name is **filename.txt**. The format of the file should be same as weights.txt explained later. 

Sample instructions are given   
Sample data is also given which is created from MNIST database   

In the data-input file, each data-point occur in each line. The values are separated by space.   
In the data-output file, the value of all the output nodes( **ideally 0 for the all the incorrect labels and 1 for the correct label** ) are given. *All numbers should be between 0 and 1 ( limits inclusive )*  

The weight matrix file contains each row in one line and in between two matrices, they are separated by two lines. All the rows of one matrix are consecutive. 

The results_test.txt files contains the values of the output nodes in each line with the values of each node space separated.The values are of the test data performed by NN_general_test.py

The confusion matrix file stores the confusion matrix with x-axis as predicted and y-axis actual values.

### The weight matrix is that matrix for which minimum error has occured on the training set which is also displayed at the last as minimum error
### During calculation of cost function , their value is just substracted from the given value which necessarily need not to be zero or one. Then apply Back-Prop Algorithm. During prediction , the one with the highest value among the output nodes is taken in all the cases both in the prediction and actual given values. The confusion matrix is also created on basis of this.
### Learning rate during entire training is kept constant.
