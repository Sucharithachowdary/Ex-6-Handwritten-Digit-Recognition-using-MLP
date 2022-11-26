# Ex-6-Handwritten Digit Recognition using MLP
## Aim:
       To Recognize the Handwritten Digits using Multilayer perceptron.
##  EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## Theory:
/*Describe here*/


## Algorithm :
1.Import the necessary libraries of python.

2.In the end_to_end function, first calculate the similarity between the inputs and the peaks. Then, to find w used the equation Aw= Y in matrix form. Each row of A (shape: (4, 2)) consists of

3.index[0]: similarity of point with peak1 index[1]: similarity of point with peak2 index[2]: Bias input (1) Y: Output associated with the input (shape: (4, )) W is calculated using the same equation we use to solve linear regression using a closed solution (normal equation).

4.This part is the same as using a neural network architecture of 2-2-1, 2 node input (x1, x2) (input layer) 2 node (each for one peak) (hidden layer) 1 node output (output layer)

5.To find the weights for the edges to the 1-output unit. Weights associated would be: edge joining 1st node (peak1 output) to the output node edge joining 2nd node (peak2 output) to the output node bias edge

## Program:
