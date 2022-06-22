import random
import math
import matplotlib.pyplot as plt

class Helper:
    def calcError(self, calculatedValue, dataLine, isXBool):
        # If calculating X, choose the third value of the data line
        # Gather velocity data for error calculation:
        #       data[2] = velocity in X
        #       data[3] = velocity in Y
        # Return errors in array:
        #       [error for RMSE, normal error]
        if isXBool:
            return (float(dataLine[2])-calculatedValue)
        return (float(dataLine[3])-calculatedValue)

    def saveWeights(self, inputLayer, hiddenLayer):
        bestInputWeightsFile = open("bestInputWeights.txt", "w+")
        bestHiddenWeightsFile = open("bestHiddenWeights.txt", "w+")

        string = ""
        for neuron in inputLayer:
            for weight in neuron.weights:
                string += str(weight)
                string += ","
            string = string[:-1]
            string += "\n"
        bestInputWeightsFile.write(string)

        string = ""
        for neuron in hiddenLayer:
            for weight in neuron.weights:
                string += str(weight)
                string += ","
            string = string[:-1]
            string += "\n"
        bestHiddenWeightsFile.write(string)

        bestInputWeightsFile.close()
        bestHiddenWeightsFile.close()

helper = Helper()

class Neuron:
    def __init__(self, weightListSize, num):
        self.id = num
        self.AV = 0
        # Create a list of weights
        list = []
        for i in range(weightListSize):
            list.append(random.random())
        self.weights = list

    def actFunction(self, landa):
        # Use Sigmoid activation model
        self.AV = 1/(1+(math.e**(-1 * self.multWeight * landa)))

    def weightMultiply(self, prevList, isHiddenBool):
        self.multWeight = 0
        current = self.id
        if isHiddenBool:
            current = self.id-1
        for node in prevList:
            # Access the activation value of a neuron from the past layer and multiply it 
            #       times the weight that relates the calling neuron (self) to that neuron
            #       in the past layer.
            res = node.AV * node.weights[current]
            self.multWeight += res

class NeuralNetwork:
    def __init__(self, nHidden, lParam, momentumParam, learnRateParam, maxNumEpochs, patience):
        # Number of neurons in the input and output neurons.
        self.NUM_INPUT = 2
        self.NUM_OUTPUT = 2
        self.MAX_EPOCHS = maxNumEpochs
        self.PATIENCE = patience

        ######          LAYER INITIALIZATION            ######
        # Initialize input layer with extra neuron for bias.
        self.inputLayer = []
        for i in range(self.NUM_INPUT+1):
            self.inputLayer.append(Neuron(nHidden, i))
        # Set value of bias for input layer
        self.inputLayer[0].AV = 1
        # Initialize hidden layer with extra neuron for bias.
        self.hiddenLayer = []
        for i in range(nHidden+1):
            self.hiddenLayer.append(Neuron(self.NUM_OUTPUT, i))
        # Set value of bias for hidden layer.
        self.hiddenLayer[0].AV = 1
        # Initialize output layer.
        self.outputLayer = []
        for i in range(self.NUM_OUTPUT):
            self.outputLayer.append(Neuron(1, i))

        ######      DELTA WEIGHT LISTS INITIALIZATION      ######
        ### This lists are going to be used with alfa parameter for momentum calculations.
        # Deltas for weights between output and hidden layers
        self.deltaWList = [[0]*self.NUM_OUTPUT for i in range(nHidden+1)]
        # Deltas for weights between input and hidden layers
        self.deltaWHList = [[0]*nHidden for i in range(self.NUM_INPUT + 1)]
        
        ######          GRADIENT INITIALIZATION            ######
        # Initialize gradients with dummy values since they will be eventually updated.
        self.hiddenGradients = [0] * nHidden
        self.outputGradients = [0] * self.NUM_OUTPUT

        ######          ERROR RELATED INITIALIZATION          ######
        # Initialize the error list to keep track of errors.
        self.errorList = [0] * self.NUM_OUTPUT
        # Initialize sum of error that will be used to calculate RMSE.
        self.errorSquaredSumXTraining = 0
        self.errorSquaredSumYTraining = 0
        self.errorSquaredSumXValidation = 0
        self.errorSquaredSumYValidation = 0

        ######          CONSTANT INITIALIZATION            ######
        # Initialize landa, alfa and eta constants with passed parameters.
        self.landa = lParam
        self.alfa = momentumParam
        self.eta = learnRateParam

        ######              OPEN DATA FILES                 ######
        # Training data file
        trainingFile = open(r"NewTrainingData.csv")
        self.trainingDataList= []
        for line in trainingFile:
            self.trainingDataList.append(line.rstrip().split(","))
        trainingFile.close()
        # Validation data file
        validationDataFile = open(r"NewValidationData.csv")
        self.validationDataList = []
        for line in validationDataFile:
            self.validationDataList.append(line.rstrip().split(","))
        validationDataFile.close()

    def forwardProcessing(self, data, isLearningBool, isPlayingBool):
        # Gather distance to target data for Input Layer:
        #       data[0] = distance in X
        #       data[1] = distance in Y
        self.inputLayer[1].AV = float(data[0])
        self.inputLayer[2].AV = float(data[1])
        # print("#### FORWARD PROCESSING ####")
        # print(" - Hidden:")
        # Hidden layer activation
        for node in self.hiddenLayer:
            if node.id != 0:
                node.weightMultiply(self.inputLayer, True)
                node.actFunction(self.landa)
                # print("         Neuron #"+str(node.id)+":    Sum="+str(node.multWeight)+" // Av="+str(node.AV))

        # print(" - Output:")
        # Output layer activation
        for node in self.outputLayer:
            node.weightMultiply(self.hiddenLayer, False)
            node.actFunction(self.landa)
            # print("         Neuron #"+str(node.id)+":    Sum="+str(node.multWeight)+" // Av="+str(node.AV))

        if not isPlayingBool:
            if isLearningBool:
                self.errorList[0] = helper.calcError(self.outputLayer[0].AV, data, True)
                self.errorSquaredSumXTraining += (self.errorList[0] * self.errorList[0])
                self.errorList[1] = helper.calcError(self.outputLayer[1].AV, data, False)
                self.errorSquaredSumYTraining += (self.errorList[1] * self.errorList[1])
                # print(" - Errors:   ErrorX="+str(self.errorList[0])+" // ErrorY="+str(self.errorList[1]))
            else:
                validationErrorX = helper.calcError(self.outputLayer[0].AV, data, True)
                validationErrorY = helper.calcError(self.outputLayer[1].AV, data, False)
                self.errorSquaredSumXValidation += (validationErrorX * validationErrorX)
                self.errorSquaredSumYValidation += (validationErrorY * validationErrorY)

    def backpropagation(self):
        # Calculate local gradients for output neurons
        # print("\n#### BACKWARD PROPAGATION ####")
        # print(" - Output gradients:")
        for i in range(self.NUM_OUTPUT):
            y = self.outputLayer[i].AV
            self.outputGradients[i] = self.landa * y * (1 - y) * self.errorList[i]
        #     print("     OutputGradient " + str(i)+": " + str(self.outputGradients[i]))
        # print(" - Hidden gradients:")
        # Calculate local gradients for hidden neurons
        for n in range(1, len(self.hiddenLayer)):
            node = self.hiddenLayer[n]
            sum = 0
            # Define sum of multiplications of local gradients and their related weights between hidden 
            #       and output layer.
            for w in range(self.NUM_OUTPUT):  
                sum += node.weights[w] * self.outputGradients[w]
            self.hiddenGradients[n-1] = self.landa * node.AV * (1 - node.AV) * sum
            # print("     HiddenGradient " + str(n)+": " + str(self.hiddenGradients[n-1]))

        # print(" - Output deltas:")
        ######      OUTPUT-HIDDEN WEIGHTS UPDATE      ######
        # Back-propagate from output layer to hidden layer calculating deltas & changing all weights 
        # for each hidden neuron
        for node in self.hiddenLayer:
            for i in range(self.NUM_OUTPUT):
                momentum =  self.alfa * self.deltaWList[node.id][i]
                # Calculate delta weight and update in matrix used for momentum calculations
                self.deltaWList[node.id][i] = self.eta * self.outputGradients[i] * node.AV + momentum
                # Update weight in neuron list
                node.weights[i] += self.deltaWList[node.id][i]
                # print("     Weight "+str(node.id)+"~"+str(i)+":     Delta="+str(self.deltaWList[node.id][i])+" // NW="+str(node.weights[i]))

        # print(" - Hidden deltas:")
        ######      HIDDEN-INPUT WEIGHTS UPDATE      ######
        # Back-propagate from hidden layer to output layer calculating deltas & changing the 4 
        # weights for each input neuron
        for node in self.inputLayer:
            for i in range(1, len(self.hiddenLayer)):
                momentum =  self.alfa * self.deltaWHList[node.id][i-1]
                # Calculate delta weight and update in matrix used for momentum calculations
                self.deltaWHList[node.id][i-1] = self.eta * self.hiddenGradients[i-1] * node.AV + momentum
                # Update weight in neuron list
                node.weights[i-1] += self.deltaWHList[node.id][i-1]
                # print("     Weight "+str(node.id)+"~"+str(i-1)+":     Delta="+str(self.deltaWHList[node.id][i-1])+" // NW="+str(node.weights[i-1]))

    def calculateRMSE(self, isLearningBool):
        # Check stage to use appropiate sums of errors
        if isLearningBool:
            # Learning and not yet validating
            rmseX = math.sqrt((self.errorSquaredSumXTraining/len(self.trainingDataList)))
            rmseY = math.sqrt((self.errorSquaredSumYTraining/len(self.trainingDataList)))
            self.errorSquaredSumXTraining = 0
            self.errorSquaredSumYTraining = 0
        else:
            # Validating
            rmseX = math.sqrt((self.errorSquaredSumXValidation/len(self.validationDataList)))
            rmseY = math.sqrt((self.errorSquaredSumYValidation/len(self.validationDataList)))
            self.errorSquaredSumXValidation = 0
            self.errorSquaredSumYValidation = 0
        # Return average of RMSE-X and RMSE-Y
        return (rmseX + rmseY)/2

    def validate(self):
        # Run only forward propagation to measure performance of network up to the 
        #       current epoch
        for data in self.validationDataList:
            self.forwardProcessing(data, False, False)
        return self.calculateRMSE(False)

    def learn(self):
        # Run forward and backward propagation and calculate RMSE
        for data in self.trainingDataList:
            self.forwardProcessing(data, True, False)
            self.backpropagation()
        return self.calculateRMSE(True)

    def train(self):
        # Train first epoch to compare after first iteration insider of the loop
        lastLearningRMSE = self.learn()
        lastValidatingRMSE= self.validate()
        # Create and append RMSE values to arrays to plot at the end
        validatingRMSEList = []
        validatingRMSEList.append(lastValidatingRMSE)
        learningRMSEList = []
        learningRMSEList.append(lastLearningRMSE)
        # print("- Epoch #1: Learning RMSE: " + str(lastLearningRMSE) + " // Validating RMSE: " + str(lastValidatingRMSE))
        # Training will stop after the validation RMSE increases for 5 straight epochs
        # This margin is given to try and avoid stopping in a local minimum
        stopAfterIncreasingErrors = self.PATIENCE
        numberOfEpochs = 1
        bestPerformance = lastValidatingRMSE
        while stopAfterIncreasingErrors > 0 and numberOfEpochs < self.MAX_EPOCHS:
            numberOfEpochs += 1
            # print("    || Learning")
            currentLearningRMSE = self.learn()
            learningRMSEList.append(currentLearningRMSE)
            # print("    || Validating")
            currentValidatingRMSE = self.validate()
            validatingRMSEList.append(currentValidatingRMSE)
            # print("- Epoch #"+str(numberOfEpochs)+": Learning RMSE: "+str(currentLearningRMSE)+" // Validating RMSE: "+str(currentValidatingRMSE))
            if (currentValidatingRMSE > lastValidatingRMSE):
                # print("                            INCREASING")
                stopAfterIncreasingErrors -= 1
            else:
                # print("                            DECREASING")
                # Save current weights if the best performance was acheived
                if currentValidatingRMSE < bestPerformance:
                    bestPerformance = currentValidatingRMSE
                    # Write/overwrite the files that store the "optimal" weights
                    helper.saveWeights(self.inputLayer, self.hiddenLayer)
                stopAfterIncreasingErrors = self.PATIENCE
            # Update values for comparisons
            lastValidatingRMSE = currentValidatingRMSE
            lastLearningRMSE = currentLearningRMSE
            # Shuffle order of rows in both training and validating value to avoid
            #       the network learning the pattern.
            random.shuffle(self.trainingDataList)
            random.shuffle(self.validationDataList)

        # print("\nTraining finished after " + str(numberOfEpochs) + " epochs.")
        # print("Best performance achieved: " + str(bestPerformance))

        # Plot behavior of both RMSEs related to the number of epochs
        # xAxisEpochs = list(range(1,numberOfEpochs+1))
        # Plot validating and learning RMSEs
        # plt.plot(xAxisEpochs, validatingRMSEList, color='r', label='Validation')
        # plt.plot(xAxisEpochs, learningRMSEList, color='b', label='Learning')
        # Add labels to both axis
        # plt.xlabel("Number of epochs")
        # plt.ylabel("RMSE")
        # Add legend
        # plt.legend()
        # Show plots
        # plt.show()

    def loadBestWeights(self):
        bestHOW = open(r"M:\NeuralNetworks\Lab\Lab2\bestHiddenWeights.txt")
        bestIHW = open(r"M:\NeuralNetworks\Lab\Lab2\bestInputWeights.txt")

        self.hiddenWeights = []
        for line in bestHOW:
            line = line.rstrip().split(",")
            for x in range(len(line)):
                line[x] = float(line[x])
            self.hiddenWeights.append(line)

        self.inputWeights = []
        for line in bestIHW:
            line = line.rstrip().split(",")
            for x in range(len(line)):
                line[x] = float(line[x])
            self.inputWeights.append(line)

        bestHOW.close()
        bestIHW.close()

        for i in range(len(self.inputLayer)):
            self.inputLayer[i].weights = self.inputWeights[i]
        for i in range(len(self.hiddenLayer)):
            self.hiddenLayer[i].weights = self.hiddenWeights[i]
