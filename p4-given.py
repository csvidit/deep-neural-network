'''
Starter code author: Steven Bogaerts
'''

import numpy as np

'''
A deep neural network classifier.
The "public" methods are forward, fit, predict, and reportParams.
'''
class DeepNNClassifier:
    '''
    layerIDToNumUnits is a list of number of units at each layer [inputUnits, layer1, layer2, ..., layerL]
    Example: [4, 4, 3, 1] means [4 inputs, 4 hidden1, 3 hidden2, 1 output], so L is 3
    
    print_cost should be True (to print costs periodically, in training) or False
    
    It is important that you are aware of these fields, as you will need them in your own code.
    '''
    def __init__(self, layerIDToNumUnits, learning_rate, num_epochs, print_cost):
        self.layerIDToNumUnits = layerIDToNumUnits # see comment above
        self.L = len(self.layerIDToNumUnits)-1   # index of the output layer
        self.nL = self.layerIDToNumUnits[self.L] # number of units in last layer
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.print_cost = print_cost
        self.costs = [] # We'll record the cost of the most recent fit, every 100 epochs

    '''
    This method initializes the parameters of the network: W and b for each layer.
    Note the use of a dictionary, self.params, to contain all of these parameters.
    
    You may recall that a dictionary is the same as a hash table or hash map.
    It maps a key (here, a string representing a parameter name) 
    to a value (here, a numpy matrix).
    
    For example, I could say:
        self.params["fakeParam"] = np.array([[1],[2],[3]])
    and then later access it with self.params["fakeParam"]
    
    In the params dictionary, each key is a variable name, like “W2” for the 
    weight matrix of layer 2, or “db3” for the db matrix of layer 3.
    
    So we're using a dictionary as a convenient way to organize our numerous parameters
    into a single structure, self.params
    
    The W and b matrices for each layer are initialized below.
    In other methods, you'll put other parameters into self.params as needed,
    and access them as needed.
    
    It is very important that you understand the use of string concatenation to 
    select a particular parameter, as shown on the left hand side of the assignment
    statements below. Note the range of the loop as well. In some other methods, you'll
    need slightly different ranges.
    '''
    def __initParams(self):
        self.params = {}
        rng = np.random.default_rng(42)
        for i in range(1, self.L+1):  # So i takes on each value in [1, 2, ..., self.L]
            # This multiplier gives us what is called "He initialization" for each weight matrix,
            # named after its creator/discoverer.
            multiplier = np.sqrt(2/self.layerIDToNumUnits[i-1])
            
            self.params["W" + str(i)] = rng.standard_normal((self.layerIDToNumUnits[i], self.layerIDToNumUnits[i-1])) * multiplier # Note the required shapes!
            self.params["b" + str(i)] = np.zeros((self.layerIDToNumUnits[i], 1)) # Make a matrix of the given shape, of all 0's

    '''
    As another illustration of the use of the params dictionary, note the code
    below for updating the parameters. Observe how this code implements the update
    steps for backpropagation, as shown in blue in the pseudocode.
    Of course, it requires that the params field already has the correct dW and db 
    values for each layer.
    '''
    def __updateParams(self):
        for layerID in range(1, self.L+1):
            sL = str(layerID)
            self.params["W"+sL] = self.params["W"+sL] - self.learning_rate * self.params["dW"+sL]
            self.params["b"+sL] = self.params["b"+sL] - self.learning_rate * self.params["db"+sL]
    
    '''
    This method prints the values of all parameters in a rather easy-to-read manner.
    It uses "..." to show when there are additional values not getting printed.
    We can adjust how many values get printed in each row and column via the optional maxDim parameter.
    If you want to skip understanding this method's implementation, that's fine, but be sure to use it.
    '''
    def reportParams(self, maxDim=5):
        print("##### reportParams #####")
        for k, v in self.params.items():
            print("----------")
            print(k, v.shape)
            
            rBound = min(maxDim, v.shape[0])
            for r in range(rBound):
                
                cBound = min(maxDim, v.shape[1])
                for c in range(cBound):
                    print("{:12.8f}".format(v[r,c]), "  ", end="")
                    
                if (cBound < v.shape[1]):
                    print("...")
                else:
                    print()
                    
            if (rBound < v.shape[0]):
                print("...\n")
            else:
                print()
        print("##### end of reportParams #####")
        
    '''
    This code performs assertions (via a helper method) to ensure that, for each 
    parameter, the derivative has the same shape as the original parameter.
    Since shape errors are common in attempts to implement a deep neural network, 
    it will be important that you call this method as you test your code.
    It will intentionally cause an error if certain dimensions don't match.
    (That's what an assertion is: "make sure this is true, or throw an error")
     
    Passing the tests here doesn't guarantee your code is correct, but it can
    help catch some common errors in W, b, dW, and db shapes.
    '''
    def __assertShapes(self):
        for layerID in range(1, self.L+1):
            sL = str(layerID)
            self.__assertOneShape("W"+sL)
            self.__assertOneShape("b"+sL)
            
    '''
    This code performs an assertion to ensure that the derivative of a structure
    has the same shape as the original structure. It is called repeatedly by the
    __assertShapes method.
    '''
    def __assertOneShape(self, s):
        assert(self.params["d"+s].shape == self.params[s].shape)
        
    # ========================================================================
    # ========================================================================
    # ========================================================================
    # ========================================================================
    # TO DO methods
    # ========================================================================
    
    '''
    (Given code - nothing more to write here)
    Computes the sigmoid of a given value, with a little bit of extra work
    for numerical stability (avoiding small numbers being rounded to 0, which
    would result in /0.)
    '''
    def __sigmoid(self, z):
        # Avoid 0 and 1, for numerical stability with floating point precision
        epsilon = 1.0e-12
        result = 1/(1+np.exp(-z) + epsilon) + epsilon
        return result
    
    '''
    TO DO
    Takes a Z matrix and returns the matrix resulting from applying the derivative 
    of the sigmoid function to each element in Z.
    
    HINT: If you're tempted to google "derivative of a Python function", you're 
    on the wrong track. Check your notes to remind yourself what this is.
    '''    
    def __sigmoidPrime(self, Z):
        vecS = np.vectorize(DeepNNClassifier.__sigmoid)
        spZ = vecS(self, Z)*(1-vecS(self, Z))
        return spZ
    
    '''
    TO DO
    Takes a Z matrix and returns the matrix resulting from applying the ReLU function 
    to each element in Z.
    '''
    
    def __max(self, z):
        return max(0.0,z)
    
    def __relu(self, Z):
        vecM = np.vectorize(DeepNNClassifier.__max)
        return vecM(self, Z)
    
    '''
    TO DO
    Takes a Z matrix and returns the matrix resulting from applying the derivative 
    of the ReLU function to each element in Z.
    '''
    
    def __rpHelper(self, z):
        if z<0.0:
            return 0.0
        else:
            return 1.0
    
    def __reluPrime(self, Z):
        vecRP = np.vectorize(DeepNNClassifier.__rpHelper)
        return vecRP(self, Z)
    
    '''
    TO DO
    layerID is the ID of a layer (between 1 and L, inclusive).
    activationFunc is a function representing the activation function
    to apply to this layer's Z matrix.

    Computes the forward activations at the given layer.
    It assumes that the previous layer's A matrix is set already (in self.params),
    and will in turn set self.params values for the given layer's Z and A matrices.
    It will also need to use the given layer's W and b matrices (in self.params),
    which should also be set already.
    '''
    def __forwardOneLayer(self, layerID, activationFunc):
        self.params["Z" + str(layerID)] = np.dot(self.params["W" + str(layerID)], self.params["A" + str((layerID-1))]) + self.params["b" + str(layerID)]
        self.params["A" + str(layerID)] = activationFunc(self.params["Z" + str(layerID)])
        
    '''
    TO DO
    For the given input activation matrix X, this method returns the activation 
    of the output unit. Use ReLU for the hidden layers, and sigmoid for the output 
    layer. You'll need to call the __forwardOneLayer method repeatedly.
    
    Remember, if you want to pass a function (like relu) into another function,
    you need to make use of Python's first-class functions property. You can *call*
    the relu function with:
        self.__relu(some matrix)
    but if you want to refer to the function itself, you'd instead use:
        self.__relu
    with no parentheses!
    '''
    def forward(self, X):
        self.params["A0"] = X
        for i in range(1, self.L):
            self.__forwardOneLayer( i, self.__relu)
        self.__forwardOneLayer( self.L, self.__sigmoid)
        return self.params["A" + str(self.L)]
    
    '''
    TO DO
    layerID is a layerID (between 1 and L, inclusive).
    activationFuncPrime is a function representing the derivative of the 
    activation function used at that layer.
    m is the number of examples in the training set.
    
    This function computes the backward calculation at the given layer.
    It should set dZ, dW, and db at the given layer, and dA at the previous layer.
    
    This code depends on dA, Z, and W being already set for layer layerID,
    and A being already set for the previous layer.
    '''
    def __backwardOneLayer(self, layerID, activationFuncPrime, m):
        self.params["dZ" + str(layerID)] = self.params["dA" + str(layerID)] * activationFuncPrime(self.params["Z" + str(layerID)])
        self.params["dW" + str(layerID)] = (1/m) * np.dot(self.params["dZ" + str(layerID)], self.params["A" + str(layerID-1)].T)
        self.params["db" + str(layerID)] = (1/m) * np.sum(self.params["dZ" + str(layerID)], axis = 1, keepdims = True)
        self.params["dA" + str(layerID-1)] = np.dot(self.params["W"+ str(layerID)].T, self.params["dZ" + str(layerID)])
        #print(self.params["W" + str(layerID)])
        #print(self.params["b" + str(layerID)])
    '''
    TO DO
    Computes a complete backward pass.
    Y is the actual, correct answers.
    m is the number of examples in the training set.
    
    It first computes dA for layer L using the formula provided in the pseudocode.
    Then __backwardOneLayer is called from the last layer to the first.
    It uses the appropriate activation derivative (__sigmoidPrime or __reluPrime)
    depending on the layer.
    
    It is important that this code correctly backpropagates through *every* layer,
    from L to 1, inclusive. It must not skip layer L or 1, for example.
    '''
    def __backward(self, Y, m):
        
        self.params["dA" + str(self.L)] = -np.divide(Y, self.params["A" + str(self.L)]) + np.divide(1-Y, 1-self.params["A" + str(self.L)])
        self.__backwardOneLayer(self.L, self.__reluPrime, m)
        l = self.L-1
        while l >= 1:
            self.__backwardOneLayer(l, self.__sigmoidPrime, m)
            l-=1
            
    
    '''
    (Given code - nothing more to write here)
    Occasionally, calls the __computeCost method (which you'll need to write).
    Tracks the cost in the costs field (a list), and possibly prints the cost.
    '''
    def __trackCost(self, Y, networkOutputs, epochID):
        if epochID % 100 == 0: # only do this every 100th epoch
            cost = self.__computeCost(Y, networkOutputs)
            self.costs.append(cost) # record the cost
            if self.print_cost:
                print("Cost after epoch %i: %f" %(epochID, cost))
                
    '''
    TO DO
    Computes the cost by comparing Y and the networkOutputs.
    '''
    def __computeCost(self, Y, networkOutputs):
        # Fill in any other code you need here (probably no more than 1 line...)
        m = networkOutputs.shape[1]
        cost = (-1.0/m)*(np.dot(Y, np.log(networkOutputs).T) + np.dot((1-Y),np.log(1-networkOutputs).T))
        cost = cost[0][0] # The previous line will most likely make a 2-D numpy array of shape (1,1), so this gets just a single value
        return cost
    
    '''
    TO DO
    Fits the parameters to the given training set.
    The training should run for self.num_epochs epochs.
    
    For each epoch, 
    self.__trackCost is called for tracking,
    and self.__assertShapes for some basic error checking.
    
    Calls the methods defined above to implement the details.
    
    Don't forget to initialize the parameters!
    
    It is also essential that, before the network begins training,
    you reset self.costs = []. (Since the costs field is only intended
    to track the most recent fit call.)
    '''
    def fit(self, X, Y):
        
        m = X.shape[1]
        self.__initParams()
        
        for i in range(self.num_epochs):
            forX = self.forward(X)
            self.__backward(Y, m)
            self.__updateParams()
            self.__trackCost(Y, forX, i)
            self.__assertShapes()
        
    
    '''
    (Given code - nothing more to write here)
    Generates predictions for the given inputs.
    Requires implementation of the forward method to work.
    '''
    def predict(self, X):
        A = self.forward(X)                  # Do a forward propagation to get the predictions
        predictions = np.rint(A)             # Round to nearest int (0 or 1)
        return predictions
    
    #def networkSizeComparison():
        
    
    # ========================================================================
    # ========================================================================
    # ========================================================================
    # ========================================================================
    # Testing methods
    # ========================================================================
    
    @staticmethod
    def test_all():
        DeepNNClassifier.test_sigmoidPrime()
        DeepNNClassifier.test_relu()
        DeepNNClassifier.test_reluPrime()
        DeepNNClassifier.test_forwardOneLayer()
        DeepNNClassifier.test_forward()
        DeepNNClassifier.test_backwardOneLayer()
        DeepNNClassifier.test_backward()
        DeepNNClassifier.test_computeCost()
        DeepNNClassifier.test_fit()
    
    @staticmethod
    def test_sigmoidPrime():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_sigmoidPrime ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        
        Z = np.array([[0.4, 0.5, 0.8]])
        print(alg.__sigmoidPrime(Z))

    @staticmethod
    def test_relu():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_relu ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        
        Z = np.array([[-0.01, 0, 0.01]])
        print(alg.__relu(Z))

    @staticmethod
    def test_reluPrime():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_reluPrime ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        
        Z = np.array([[-0.01, 0, 0.01]])
        print(alg.__reluPrime(Z))

    @staticmethod
    def test_forwardOneLayer():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_forwardOneLayer ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        alg.__initParams()
        
        layerID = 1
        alg.params["A0"] = np.array([[0.7, 0.1, 0.2],  # 3 examples, 4 input attributes each
                                     [0.8, 0.3, 0.4], 
                                     [0.9, 0.5, 0.6], 
                                     [0.1, 0.7, 0.8]]) 
        # Of course, similar tests should work for other layerIDs (and inputs).
        
        alg.__forwardOneLayer(layerID, alg.__sigmoid) # Should mutate Z1 and A1
        
        print("Z1:", alg.params["Z1"], sep='\n', end='\n\n')
        print("A1:", alg.params["A1"], sep='\n', end='\n\n')

    @staticmethod
    def test_forward():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_forward ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        alg.__initParams()
        
        X = np.array([[0.7,  0.1,  0.2],  # 3 examples, 4 input attributes each
                      [0.8,  0.3,  0.4], 
                      [0.9,  0.5,  0.6], 
                      [0.1,  0.7,  0.8]]) 
        
        alg.forward(X)
        
        print("A0:", alg.params["A0"], sep='\n', end='\n\n')
        print("Z1:", alg.params["Z1"], sep='\n', end='\n\n')
        print("A1:", alg.params["A1"], sep='\n', end='\n\n')
        print("Z2:", alg.params["Z2"], sep='\n', end='\n\n')
        print("A2:", alg.params["A2"], sep='\n', end='\n\n')
        print("Z3:", alg.params["Z3"], sep='\n', end='\n\n')
        print("A3:", alg.params["A3"], sep='\n', end='\n\n')
        print("Z4:", alg.params["Z4"], sep='\n', end='\n\n')
        print("A4:", alg.params["A4"], sep='\n', end='\n\n')

    @staticmethod
    def test_backwardOneLayer():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_backwardOneLayer ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        alg.__initParams()
        
        layerID = 1
        m = 3
        X = np.array([[0.7,  0.1,  0.2],  # 3 examples, 4 input attributes each
                      [0.8,  0.3,  0.4], 
                      [0.9,  0.5,  0.6], 
                      [0.1,  0.7,  0.8]])
        Y = np.array([[1, 1, 0]])
        alg.forward(X) # sets all A and Z params
        alg.params["dA"+str(alg.L)] = -np.divide(Y, alg.params["A"+str(alg.L)]) + np.divide(1-Y, 1-alg.params["A"+str(alg.L)])
        
        layerID = alg.L
        print("Calling backwardOneLayer on last layer...")
        alg.__backwardOneLayer(layerID, alg.__sigmoidPrime, m)
        sL = str(layerID)
        print("dZ" + sL + ":", alg.params["dZ"+sL], sep='\n', end='\n\n')
        print("dW" + sL + ":", alg.params["dW"+sL], sep='\n', end='\n\n')
        print("db" + sL + ":", alg.params["db"+sL], sep='\n', end='\n\n')
        print("dA" + str(layerID-1) + ":", alg.params["dA"+str(layerID-1)], sep='\n', end='\n\n')
        
        print("Calling backwardOneLayer on next to last layer...")
        layerID = alg.L-1
        alg.__backwardOneLayer(alg.L-1, alg.__reluPrime, m)
        sL = str(layerID)
        print("dZ" + sL + ":", alg.params["dZ"+sL], sep='\n', end='\n\n')
        print("dW" + sL + ":", alg.params["dW"+sL], sep='\n', end='\n\n')
        print("db" + sL + ":", alg.params["db"+sL], sep='\n', end='\n\n')
        print("dA" + str(layerID-1) + ":", alg.params["dA"+str(layerID-1)], sep='\n', end='\n\n')

    @staticmethod
    def test_backward():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_backward ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        alg.__initParams()
        
        m = 3
        X = np.array([[0.7,  0.1,  0.2],  # 3 examples, 4 input attributes each
                      [0.8,  0.3,  0.4], 
                      [0.9,  0.5,  0.6], 
                      [0.1,  0.7,  0.8]])
        Y = np.array([[1, 1, 0]])
        
        alg.forward(X)
        alg.__backward(Y, m)
        
        alg.reportParams(5)
        
        print("Params keys:", alg.params.keys())

    @staticmethod
    def test_computeCost():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_computeCost ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        
        Y = np.array([[1, 1, 0]])
        networkOutputs = np.array([[.8, .2, .4]])
        print(alg.__computeCost(Y, networkOutputs))

    @staticmethod
    def test_fit():
        print("========================================")
        print("========================================")
        print("========================================")
        print("---------- test_fit ----------")
                                             #              layer l    layer 2    layer 3    layer 4=L
        alg = DeepNNClassifier([4, 5, 3, 4, 1], # 4 inputs, 5 hidden1, 3 hidden2, 4 hidden3, 1 output
                               learning_rate=0.02, num_epochs=5, print_cost=True)
        
        X = np.array([[0.7,  0.1,  0.2],  # 3 examples, 4 input attributes each
                      [0.8,  0.3,  0.4], 
                      [0.9,  0.5,  0.6], 
                      [0.1,  0.7,  0.8]])
        Y = np.array([[1, 1, 0]])
        
        alg.fit(X, Y)
        alg.reportParams(5)


import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

class DataManager:

    def __changePandasToNumpy(self, inputsDF, outputSeries):
        X = inputsDF.to_numpy().T # each column is one example
        Y = outputSeries.to_numpy().reshape(1, outputSeries.shape[0]) # row vector of outputs, as a (1, m) matrix
        return X, Y

    '''
    Reads from two .csv files (___Train.csv, and ___Test.csv) with the provided filename as the root of each name.
    Returns structures for the inputs and outputs of the training and testing sets.
    
    If you'd like to see a plot of the data, pass True in for the plot parameter.
    '''
    def readDataset(self, filename, plot=False):
        # Read the training and testing sets from the files
        trainDF = pd.read_csv("data/" + filename + "Train.csv", index_col=False, na_values="", delimiter = ",")
        testDF = pd.read_csv("data/" + filename + "Test.csv", index_col=False, na_values="", delimiter = ",")
        
        # Get the name of the output attribute
        targetID = -1 # In all datasets I've provided, the last column is the output column
        target = trainDF.columns[targetID]
        
        # Optionally, plot the full dataset after reading the combined file.
        if (plot):
            allDF = pd.read_csv("data/" + filename + ".csv", index_col = False, na_values="", delimiter=",")
            self.__plotDataset(allDF, target)
        
        predictors = list(trainDF.columns)
        predictors.remove(target)
        # Now, predictors is only the input attribute names.
        # target is the output attribute name.
        
        # Convert from pandas data structures into numpy, as we've been using in our work
        X_train, Y_train = self.__changePandasToNumpy(trainDF.loc[:, predictors], trainDF.loc[:, target])
        X_test, Y_test = self.__changePandasToNumpy(testDF.loc[:, predictors], testDF.loc[:, target])
        
        return X_train, Y_train, X_test, Y_test

    '''
    Plots the given dataset.
    '''    
    def __plotDataset(self, df, targetCol):
        plt.figure()
        colorMap = {0:'r', 1:'b'}
        fig, ax = plt.subplots(1,1)
        ax.scatter(df.iloc[:, 0], df.iloc[:, 1], c=[colorMap[classification] for classification in df.iloc[:, 2]]) # https://seaborn.pydata.org/examples/wide_data_lineplot.html
        ax.set(xlabel="a", ylabel="b") # https://seaborn.pydata.org/examples/scatterplot_categorical.html
        plt.show()    

'''
Takes two (1, m) output matrices, and determines an accuracy (percentage of values that match).
Uses the accuracy_score function from sklearn.
'''
def compareResults(Y_predicted, Y_correct):
    return metrics.accuracy_score(np.squeeze(Y_predicted), np.squeeze(Y_correct))

'''
TO DO
For the given baseFilename, reads the dataset structures, makes a deep NN classifier,
and tests it.
'''
def runFullTest(baseFilename, plot=False):
    dm = DataManager()
    dm.readDataset(baseFilename, plot=False)
    if baseFilename == "linear" or baseFilename == "complex":
        alg = DeepNNClassifier([2, 4, 2, 1], learning_rate=0.2, num_epochs=10000, print_cost=True)
    else:
        alg = DeepNNClassifier([4, 4, 2, 1], learning_rate=0.2, num_epochs=10000, print_cost=True)
    X_train, Y_train, X_test, Y_test = dm.readDataset(baseFilename, plot=False)
    alg.fit(X_train, Y_train)
    compareResults(alg.predict(X_train))
    compareResults(alg.predict(X_test))

# runFullTest("linear", True)
# runFullTest("complex", True)
# runFullTest("fourInputs", False) # Not set up to plot in higher dimensions.

def networkSizeComparison():
    alg1 = DeepNNClassifier([4, 4, 1], learning_rate=0.2, num_epochs=4000, print_cost=True)
    alg2 = DeepNNClassifier([4, 4, 3, 1], learning_rate=0.2, num_epochs=4000, print_cost=True)
    alg3 = DeepNNClassifier([4, 8, 6, 4, 2, 1], learning_rate=0.2, num_epochs=4000, print_cost=True)
    
    dm = DataManager()
    X_train, Y_train, X_test, Y_test = dm.readDataset("fourInputs", True)
    print("Accuracy for TRAINING set of Network 1 : ", compareResults(alg1.predict(X_train)))
    print("Accuracy for TESTING set of Network 1 : ", compareResults(alg1.predict(X_test)))
    print("Accuracy for TRAINING set of Network 2 : ", compareResults(alg2.predict(X_train)))
    print("Accuracy for TESTING set of Network 2 : ", compareResults(alg2.predict(X_test)))
    print("Accuracy for TRAINING set of Network 2 : ", compareResults(alg3.predict(X_train)))
    print("Accuracy for TESTING set of Network 2 : ", compareResults(alg3.predict(X_test)))
    
def alphaComparison():
    alg1 = DeepNNClassifier([2, 5, 4, 1], learning_rate=0.01, num_epochs=10000, print_cost=True)
    alg2 = DeepNNClassifier([2, 5, 4, 1], learning_rate=0.03, num_epochs=10000, print_cost=True)
    alg3 = DeepNNClassifier([2, 5, 4, 1], learning_rate=0.1, num_epochs=10000, print_cost=True)
    alg4 = DeepNNClassifier([2, 5, 4, 1], learning_rate=0.3, num_epochs=10000, print_cost=True)
    alg5 = DeepNNClassifier([2, 5, 4, 1], learning_rate=1, num_epochs=10000, print_cost=True)
    alg6 = DeepNNClassifier([2, 5, 4, 1], learning_rate=3, num_epochs=10000, print_cost=True)
    
def unitsAndLayers():
    alg1 = DeepNNClassifier([4, 16, 1], learning_rate=0.1, num_epochs=5000, print_cost=True)
    alg2 = DeepNNClassifier([4, 12, 4, 1], learning_rate=0.1, num_epochs=5000, print_cost=True)
    alg3 = DeepNNClassifier([4, 8, 6, 2, 1], learning_rate=0.1, num_epochs=5000, print_cost=True)
    alg4 = DeepNNClassifier([4, 7, 4, 3, 2, 1], learning_rate=0.1, num_epochs=5000, print_cost=True)
    alg5 = DeepNNClassifier([4, 4, 4, 4, 4, 1], learning_rate=0.1, num_epochs=5000, print_cost=True)
    alg6 = DeepNNClassifier([4, 6, 4, 3, 2, 1, 1], learning_rate=0.1, num_epochs=5000, print_cost=True)
    
# -----------------------------------------------------------------
# Exercise 5

import seaborn as sns

'''
Makes line plots of costs for networks with the given labels, across many epochs.

costs is a list of lists (a 2-D list), representing the cost every 100 epochs
(since that's what __trackCosts tracks)
for each network under consideration.
For example, if there are just two networks under consideration,
and the first network tracked costs 0.6, 0.55, 0.54, 
   (at epochs 0, 100, and 200, since __trackCosts tracks every 100 epochs)
and the second tracked costs 0.59, 0.57, 0.53,
   (again, at epochs 0, 100, and 200)
then the costs list should be [[0.6, 0.55, 0.54], [0.59, 0.57, 0.53]].

labels is a list of the data labels.
For example, for problem (5), alphaComparison, labels should be a list of the alpha values used.
For example, if just two networks were made, the first with alpha=0.1, and 
the second with alpha=0.3, then alphas should be [0.1, 0.3].
Continuing the example from above, the network with alpha=0.1 had costs 0.6, 0.55, 0.54,
... etc.

num_epochs is an int for the number of epochs of training each network had.
For example, if each network was trained for 1000 epochs, then
num_epochs should be 1000.
Ha ha, you didn't need an example for num_epochs, I know. But aren't I hilarious.
'''
def plotCosts(costs, labels, num_epochs):
    # Adapted from https://seaborn.pydata.org/examples/wide_data_lineplot.html
    costs = np.array(costs).T # convert to columns of costs (1 column corresponds to one alpha value)
    epochs = np.arange(0, num_epochs, 100) # array from 0 to num_epochs, step 100 - the epochs for which we tracked costs
    ax = sns.lineplot(data=pd.DataFrame(costs, epochs, columns=[str(lab) for lab in labels]))
    ax.set(xlabel="Epoch", ylabel="Cost")
