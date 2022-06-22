# Multi-layer perceptron neural network with a Genetic Algorithm
Multi-layer perceptron neural network whose meta-parameters were optimized using a genetic algorithm. Data for training and testing was gathered in a space lander game. The NN was also tested to automatically play the space lander game.

[__DebugClassGame.py__]([url](https://github.com/abundis28/Multilayer_Perceptron_NN_GA/blob/main/DebugClassesGame.py)) contains the code with the neuron and neural network classes along with their respective functions (e.g. feedforward, backpropagation, training, etc...).

[__NeuralNetHolder.py__]([url](https://github.com/abundis28/Multilayer_Perceptron_NN_GA/blob/main/NeuralNetHolder.py)) contains the code necessary to integrate the trained neural network with the rocket game used to test. The game is not included in this file because it is not of my authorship.

[__TrainingFinal.csv__]([url](https://github.com/abundis28/Multilayer_Perceptron_NN_GA/blob/main/TrainingFinal.csv)) and [__ValidationFinal.csv__]([url](https://github.com/abundis28/Multilayer_Perceptron_NN_GA/blob/main/ValidationFinal.csv)) contain the training and validation pre-processed datasets. 

[__bestHiddenWeights.txt__]([url](https://github.com/abundis28/Multilayer_Perceptron_NN_GA/blob/main/bestHiddenWeights.txt)) and [__bestInputWeights.txt__]([url](https://github.com/abundis28/Multilayer_Perceptron_NN_GA/blob/main/bestInputWeights.txt)) contain the weights obtained after training for each of the respective layers.
