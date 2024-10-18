
This project was created for Neural network generation project,the project's overall concept and how 
it relates to this repository will be discussed below.

# Neural network generation:
A neural network takes as input a datapoint in n-dimensional space and outputs a m-dimensional output.
This description holds for any type of input, output, or neural network architecture, considering that a 
A neural network is a set of weights and biases; we can say that a neural network can be an input to 
another model or even the output.

Realizing that neural networks can be treated as datapoints opens the possibility for 
trainingless artificial intelligence, just like a classification model can classify new data points.
The same thing can be said for a model that generates neural networks. 
(It can generate a performing model that it didn't come across before.).


# Mixable dataset
This idea rests on the capacity to create datasets of trained neural networks; such a thing requires 
a large volume of base datapoints, taking classification as a case study, we require thousands 
of classification models to train a meta model that generates classifires, I opted for binary 
classification in order to have a larger set of unique models and the maximum number of 
of classes, I mixed a set of already existing datasets; to do so, I implemented MixableDataset
that takes multiple datasets (with multiple classes) as input and returns a set of binary 
classification dataset (call get_subDataset).



# Serializable models
The second component of this project is the ability to transform models into datapoints in 
In order to create the meta dataset, this requires the serialization of the models; given a neural network 
The serialization returns a vector of all its weights and biases; this functionality is implemented in the 
SerMod class.
