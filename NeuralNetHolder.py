import DebugClassesGame
class NeuralNetHolder:

    def __init__(self):
        super().__init__()
        self.nn = DebugClassesGame.NeuralNetwork(8, 0.8, 0.05, 0.8, 100, 10)
        self.nn.loadBestWeights()

    def normalize(self, input, xDescrip, yDescrip):
        xNorm = (input[0]-xDescrip[1])/(xDescrip[0]-xDescrip[1])
        yNorm = (input[1]-yDescrip[1])/(yDescrip[0]-yDescrip[1])
        return xNorm, yNorm

    def denormalize(self, output, xDescrip, yDescrip):
        xDenorm = output[0]*(xDescrip[0]-xDescrip[1]) + xDescrip[1]
        yDenorm = output[1]*(yDescrip[0]-yDescrip[1]) + yDescrip[1]
        return xDenorm, yDenorm

    def predict(self, input_row):
        # WRITE CODE TO PROCESS INPUT ROW AND PREDICT X_Velocity and Y_Velocity
        xDescripIn = [636.9667374, -641.1548365]
        yDescripIn = [511.8389584, 65.60250444]
        xDescripOut = [7.710915, -3.90031]
        yDescripOut = [5.231749, -7.02967]
        input_row = input_row.split(',')
        for i in range(len(input_row)):
            input_row[i] = float(input_row[i])
        print("- Inputs: xD="+str(input_row[0])+" // yD"+str(input_row[1]))
        xNorm, yNorm = self.normalize(input_row, xDescripIn, yDescripIn)
        self.nn.forwardProcessing([xNorm, yNorm], False, True)
        xDenorm, yDenorm = self.denormalize([self.nn.outputLayer[0].AV, self.nn.outputLayer[1].AV], xDescripOut, yDescripOut)
        print("     Denorm: xV="+str(xDenorm)+" // yV="+str(yDenorm))
        return [xDenorm, yDenorm]