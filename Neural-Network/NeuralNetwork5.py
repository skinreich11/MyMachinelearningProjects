import copy
import math
import random as random
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# To run my algorithm, just run the Python program provided. It will first print to console all the information required
# by backprop_example1 and backprop_example2, as described in those files. After printing to console the
# information above, it will train and test each of the 3 datasets, house of Votes, Wine, and Extra Credit Cancer.
# It will print to console first the epoch of the training of each fold-test pair out of the 10 folds, and then
# print the average accuracy and F1 scores across 10 folds. It will then perform and show the graph of the error curve,
# using a 70-30 training-testing split. To continue the algorithm, close the current graph window.
# I implemented a vectorized version of backpropagation. I used both combination criteria, stopping training
# the network if I presented the training data more than 1000 times, or when the difference between the previous cost
# and the current cost is less than e^-8. The weights are initialized based on a Gaussian distribution with zero mean
# and variance equal to 1, and the biases are initialized to 0. After various tests, I found that using a
# small neural network with less number of hidden layers with a larger amount of neurons per hidden layer produces
# the best results. Also, the best results were shown when the regulation parameter was between 0.0001 and 0.00001,
# as well as a step size between 0.1 and 1. I got these final six neural networks, The first parameter is the layers,
# the second is the regulation value, and the third is the step size, as shown in the PREDETERMINED_VALUES varaible.
# To the layers, I inserted to the front the input layer, which equals the number of measurements in the CSV file.
# I appended to the end the output layer, which is the number of classes in the CSV file if there are more
# than two classes, otherwise 1.

PREDETERMINED_VALUES = [
    [[2, 4], 0.0001, 1],
    [[4, 16], 0.0001, 0.1],
    [[16], 0.00001, 0.1],
    [[8, 2], 0.0001, 1],
    [[8], 0.00001, 1],
    [[4, 4, 16], 0.001, 1]
]

def highestElem(pred):
    result = []
    curHigh = 0
    for res in pred:
        if res[0] > curHigh:
            curHigh = res[0]
    result.append([curHigh])
    return np.array(result)

def fitData(array):
    maxNum = np.max(array, axis=0)
    minNum = np.min(array, axis=0)
    result = np.zeros_like(array)
    for m in range(len(array)):
        for n in range(len(array[m])):
            result[m][n] = (array[m][n] - minNum[n]) / (maxNum[n] - minNum[n])
    return result

def calcDataResults(results):
    resAccuracy = []
    for foldRes in results:
        counter = 0
        for allPredictions in foldRes:
            if allPredictions[0] == allPredictions[1]:
                counter += 1
        resAccuracy.append(counter / len(foldRes))
    resPrecision = []
    resRecall = []
    resF1 = []
    for foldRes in results:
        curPrecision = []
        curRecall = []
        for classes in unClasses:
            TP = 0
            FP = 0
            FN = 0
            for allPredictions in foldRes:
                if allPredictions[0] == classes and allPredictions[1] == classes:
                    TP += 1
                elif allPredictions[0] != classes and allPredictions[1] == classes:
                    FP += 1
                elif allPredictions[0] == classes and allPredictions[1] != classes:
                    FN += 1
            if TP+FP == 0:
                curPrecision.append(0)
            else:
                curPrecision.append(TP/(TP+FP))
            if TP+FN == 0:
                curRecall.append(0)
            else:
                curRecall.append(TP/(TP+FN))
        resPrecision.append(sum(curPrecision) / len(curPrecision))
        resRecall.append(sum(curRecall) / len(curRecall))
    for res in range(len(resPrecision)):
        resF1.append(2*((resPrecision[res]*resRecall[res])/(resPrecision[res]+resRecall[res])))
    return sum(resAccuracy) / len(resAccuracy), sum(resPrecision) / len(resPrecision), \
           sum(resRecall) / len(resRecall), sum(resF1) / len(resF1)

def create10Folds(measures, classes, unClasses, percent):
    resFolds = []
    for l in range(10):
        curFoldMeasures = []
        curFoldClasses = []
        for j in range(len(unClasses)):
            counter = 0
            while counter < int(percent[j] / 10) and len(classes) != 0:
                ind = np.argmax(classes == unClasses[j])
                if ind != -1:
                    popMeasure = measures[ind]
                    popClass = classes[ind]
                    if ind == 0:
                        measures = measures[1:]
                        classes = classes[1:]
                    elif ind == len(classes) - 1:
                        measures = measures[:-1]
                        classes = classes[:-1]
                    else:
                        measures = np.concatenate((measures[:ind], measures[ind+1:]))
                        classes = np.concatenate((classes[:ind], classes[ind+1:]))
                    curFoldMeasures.append(popMeasure)
                    curFoldClasses.append(popClass)
                else:
                    break
                counter += 1
        resFolds.append([curFoldMeasures, curFoldClasses])
    for l in range(len(classes)):
        ind = random.randint(0, 9)
        popMeasure, measures = measures[-1], measures[:-1]
        popClass, classes = classes[-1], classes[:-1]
        resFolds[ind][0].append(popMeasure)
        resFolds[ind][1].append(popClass)
    return resFolds

class NeuralNetwork:
    def __init__(self, layers, printData):
        self.layers = layers
        self.printData = printData
        self.weights = [np.random.normal(loc=0.0, scale=1.0, size=(layers[i], layers[i-1])) for i in range(1, len(layers))]
        self.biases = [np.zeros((layers[i], 1)) for i in range(1, len(layers))]
        self.regularization_param = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def softMax(self, z):
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z)

    def feedforward(self, x):
        counter = 1
        a = x
        if self.printData:
            print("Forward propagating the input ", x)
        if counter == 1:
            if self.printData:
                res_print = a.copy()
                res_print = np.insert(res_print, 0, [1.00000])
                print("a1 = ", res_print)
        for g in range(len(self.weights)):
            counter += 1
            z = np.dot(self.weights[g], a) + self.biases[g]
            a = self.sigmoid(z)
            if counter != 1:
                if self.printData:
                    if g != len(self.weights) - 1:
                        res_print = a.copy()
                        res_print = np.insert(res_print, 0, [1.00000])
                        print("z" + str(counter) + " = ", z)
                        print("a" + str(counter) + " = ", res_print)
                    else:
                        print("z" + str(counter) + " = ", z)
                        print("a" + str(counter) + " = ", a)
            if g == len(self.weights) - 1:
                if self.printData:
                    print("f(x) = ", a)
        return a

    def backpropagate(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        activation = x
        activations = [x]
        zs = []
        for o in range(len(self.weights)):
            z = np.dot(self.weights[o], activation) + self.biases[o]
            zs.append(z)
            activation = self.sigmoid(z)
            activations.append(activation)
        delta = activations[-1] - y
        nabla_w[-1] = np.dot(delta, activations[-2].T)
        nabla_b[-1] = delta
        for l in range(2, len(self.layers)):
            z = zs[-l]
            sp = self.sigmoid_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            nabla_b[-l] = delta
        return nabla_w, nabla_b

    def train(self, training_data, epochs, alph, epsilon):
        prevCost = float('inf')
        for epoch in range(epochs):
            total_cost = 0
            nabla_w_acc = [np.zeros(w.shape) for w in self.weights]
            nabla_b_acc = [np.zeros(b.shape) for b in self.biases]
            for x, y in training_data:
                nabla_w, nabla_b = self.backpropagate(x, y)
                nabla_w_acc = [nw + w for nw, w in zip(nabla_w_acc, nabla_w)]
                nabla_b_acc = [nb + b for nb, b in zip(nabla_b_acc, nabla_b)]
                predicted = self.feedforward(x)
                instance_cost = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))
                total_cost += instance_cost
                if self.printData:
                    print("Computing gradients based on training instance ", x, y)
                    for i in range(len(nabla_b)):
                        print("delta", i+2, ": ", nabla_b[i])
                    for i in range(len(nabla_b)):
                        print("Gradients of Theta", i+1, "based on training instance", x, y)
                        if len(nabla_b[i]) > 1:
                            for f in range(len(nabla_b[i])):
                                print(nabla_b[i][f], nabla_w[i][f])
                        else:
                            print(nabla_b[i], nabla_w[i])
            avg_cost = total_cost / len(training_data)
            S = sum(np.sum(theta ** 2) for theta in self.weights)
            S *= (self.regularization_param / (2 * len(training_data)))
            avg_cost += S
            reg_term = [(self.regularization_param * theta) for theta in self.weights]
            nabla_w_final = [(1/len(training_data)) * (nw + reg) for nw, reg in zip(nabla_w_acc, reg_term)]
            nabla_b_final = [(1/len(training_data)) * nb for nb in nabla_b_acc]
            self.weights = [w - alph * nw for w, nw in zip(self.weights, nabla_w_final)]
            self.biases = [b - alph * nb for b, nb in zip(self.biases, nabla_b_final)]
            if self.printData:
                print("The entire training set has been processes. Computing the average (regularized) gradients:")
                for i in range(len(nabla_b_final)):
                    print("Final regularized gradients of theta", i+1)
                    if len(nabla_b_final[i]) > 1:
                        for f in range(len(nabla_b_final[i])):
                            print(nabla_b_final[i][f], nabla_w_final[i][f])
                    else:
                        print(nabla_b_final[i], nabla_w_final[i])
            if prevCost - avg_cost < epsilon:
                print(f"Stopped at epoch {epoch+1} due to convergence.")
                return
            prevCost = avg_cost
        if epochs != 1:
            print("Reached max epoch of 1000, finishing training")


print("results and data for backprop_example1")
training_data = [(np.array([[0.13000]]), np.array([[0.90000]])),
                 (np.array([[0.42000]]), np.array([[0.23000]]))]
nn = NeuralNetwork([1, 2, 1], True)
nn.biases = [np.array([[0.40000], [0.30000]]),
             np.array([[0.70000]])]
nn.weights = [np.array([[0.10000], [0.20000]]),
              np.array([[0.50000, 0.60000]])]
nn.train(training_data, 1, 0, 0)
nn.printData = False
total_cost = 0
for x, y in training_data:
    predicted = nn.feedforward(x)
    print(f"Input: {x}, Target: {y}, Predicted: {predicted}")
    instance_cost = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))
    print("Cost, J, associated with instance ", x, y, ":", instance_cost)
    total_cost += instance_cost
avg_cost = total_cost / len(training_data)
S = sum(np.sum(theta ** 2) for theta in nn.weights)
S *= (nn.regularization_param / (2 * len(training_data)))
avg_cost += S
print("Final (regularized) cost, J, based on the complete training set: ", avg_cost)


print("results and data for backprop_example2")
training_data = [(np.array([[0.32000], [0.68000]]), np.array([[0.75000], [0.98000]])),
                 (np.array([[0.83000], [0.02000]]), np.array([[0.75000], [0.28000]]))]
nn = NeuralNetwork([2, 4, 3, 2], True)
nn.weights = [np.array([[0.15000, 0.40000], [0.10000, 0.54000], [0.19000, 0.42000], [0.35000, 0.68000]]),
             np.array([[0.67000, 0.14000, 0.96000, 0.87000], [0.42000, 0.20000, 0.32000, 0.89000], [0.56000, 0.80000, 0.69000, 0.09000]]),
             np.array([[0.87000, 0.42000, 0.53000], [0.10000, 0.95000, 0.69000]])]
nn.biases = [np.array([[0.42000], [0.72000], [0.01000], [0.30000]]),
              np.array([[0.21000], [0.87000], [0.03000]]),
              np.array([[0.04000], [0.17000]])]
nn.regularization_param = 0.250
nn.train(training_data, 1, 0, 0)
nn.printData = False
total_cost = 0
for x, y in training_data:
    predicted = nn.feedforward(x)
    print(f"Input: {x}, Target: {y}, Predicted: {predicted}")
    instance_cost = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))
    print("Cost, J, associated with instance ", x, y, ":", instance_cost)
    total_cost += instance_cost
avg_cost = total_cost / len(training_data)
S = sum(np.sum(theta ** 2) for theta in nn.weights)
S *= (nn.regularization_param / (2 * len(training_data)))
avg_cost += S
print("Final (regularized) cost, J, based on the complete training set: ", avg_cost)


print("Data of House of Votes")
df = pd.read_csv('hw3_house_votes_84.csv')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X).toarray()
unClasses, counts = np.unique(Y, return_counts=True)
allFolds = create10Folds(X_encoded, Y, unClasses, counts)
for q in range(6):
    allAccuracies = []
    allPrecisions = []
    allRecalls = []
    allF1 = []
    curRes = []
    nnValues = []
    for n in range(len(allFolds)):
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for m in range(len(allFolds)):
            if n == m:
                curTestMeasure = allFolds[n][0]
                curTestClass = allFolds[n][1]
            else:
                for foldMeasure in allFolds[n][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[n][1]:
                    curTrainClass.append(foldClass)
        trainData = [(curTrainMeasure[i], np.array([curTrainClass[i]])) for i in range(len(curTrainMeasure))]
        testData = [(curTestMeasure[i], np.array([curTestClass[i]])) for i in range(len(curTestMeasure))]
        transformedTrainData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in trainData]
        transformedTestData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in testData]
        if n == 0:
            nnValues = copy.deepcopy(PREDETERMINED_VALUES[q])
            nnValues[0].insert(0, len(transformedTrainData[0][0]))
            nnValues[0].append(1)
        nn = NeuralNetwork(nnValues[0], False)
        nn.regularization_param = nnValues[1]
        nn.train(transformedTrainData, 1000, nnValues[2], math.e ** -8)
        resPredictions = []
        for x, y in transformedTestData:
            resPredictions.append([y[0][0], round(nn.feedforward(x)[0][0])])
        curRes.append(resPredictions)
    print("Now testing the House dataset with the following values: Layers = ", nnValues[0], ", regularization_param = "
        , nnValues[1], ", alpha = ", nnValues[2])
    accuracy, precision, recall, F1 = calcDataResults(curRes)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
    print(allAccuracies, allF1)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3)
trainData = [(X_train[i], np.array([y_train[i]])) for i in range(len(X_train))]
testData = [(X_test[i], np.array([y_test[i]])) for i in range(len(X_test))]
transformedTrainData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in trainData]
transformedTestData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in testData]
allErorrs = []
nnValues = copy.deepcopy(PREDETERMINED_VALUES[4])
nnValues[0].insert(0, len(transformedTrainData[0][0]))
nnValues[0].append(1)
for q in range(60, len(transformedTrainData) + 1, 60):
    nn = NeuralNetwork(nnValues[0], False)
    nn.regularization_param = nnValues[1]
    nn.train(transformedTrainData[:q], 1000, nnValues[2], math.e ** -8)
    total_cost = 0
    for x, y in transformedTestData:
        predicted = nn.feedforward(x)
        instance_cost = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))
        total_cost += instance_cost
    avg_cost = total_cost / len(transformedTrainData)
    S = sum(np.sum(theta ** 2) for theta in nn.weights)
    S *= (nn.regularization_param / (2 * len(transformedTrainData)))
    avg_cost += S
    allErorrs.append(avg_cost)
trainingNumInst = range(60, len(transformedTrainData) + 1, 60)
plt.plot(trainingNumInst, allErorrs, marker='o', linestyle='-')
plt.xlabel('Number of training instances')
plt.ylabel('Error J of testing data')
plt.title('Error J vs training instances house of votes graph')
plt.grid(True)
plt.show()


print("Data of Wine")
df = pd.read_csv('hw3_wine.csv', sep='\t')
Y = df.iloc[:, 0].values
X = df.iloc[:, 1:].values
X_encoded = np.array(fitData(X))
unClasses, counts = np.unique(Y, return_counts=True)
allFolds = create10Folds(X_encoded, Y, unClasses, counts)
for q in range(6):
    allAccuracies = []
    allPrecisions = []
    allRecalls = []
    allF1 = []
    curRes = []
    nnValues = []
    for n in range(len(allFolds)):
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for m in range(len(allFolds)):
            if n == m:
                curTestMeasure = allFolds[n][0]
                curTestClass = allFolds[n][1]
            else:
                for foldMeasure in allFolds[n][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[n][1]:
                    curTrainClass.append(foldClass)
        trainData = [(curTrainMeasure[i], np.array([curTrainClass[i]])) for i in range(len(curTrainMeasure))]
        testData = [(curTestMeasure[i], np.array([curTestClass[i]])) for i in range(len(curTestMeasure))]
        transformedTrainData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in trainData]
        transformedTestData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in testData]
        for k in range(len(transformedTrainData)):
            if transformedTrainData[k][1][0][0] == 1:
                transformedTrainData[k] = (transformedTrainData[k][0], np.array([[1.0], [0.0], [0.0]]))
            elif transformedTrainData[k][1][0][0] == 2:
                transformedTrainData[k] = (transformedTrainData[k][0], np.array([[0.0], [1.0], [0.0]]))
            else:
                transformedTrainData[k] = (transformedTrainData[k][0], np.array([[0.0], [0.0], [1.0]]))
        for k in range(len(transformedTestData)):
            if transformedTestData[k][1][0][0] == 1:
                transformedTestData[k] = (transformedTestData[k][0], np.array([[1.0], [0.0], [0.0]]))
            elif transformedTestData[k][1][0][0] == 2:
                transformedTestData[k] = (transformedTestData[k][0], np.array([[0.0], [1.0], [0.0]]))
            else:
                transformedTestData[k] = (transformedTestData[k][0], np.array([[0.0], [0.0], [1.0]]))
        if n == 0:
            nnValues = copy.deepcopy(PREDETERMINED_VALUES[q])
            nnValues[0].insert(0, len(transformedTrainData[0][0]))
            nnValues[0].append(3)
        nn = NeuralNetwork(nnValues[0], False)
        nn.regularization_param = nnValues[1]
        nn.train(transformedTrainData, 1000, nnValues[2], math.e ** -8)
        resPredictions = []
        for x, y in transformedTestData:
            predict = nn.feedforward(x)
            predict = np.argmax(predict) + 1
            resPredictions.append([np.argmax(y) + 1, predict])
        curRes.append(resPredictions)
    print("Now testing the Wine dataset with the following values: Layers = ", nnValues[0], ", regularization_param = "
          , nnValues[1], ", alpha = ", nnValues[2])
    accuracy, precision, recall, F1 = calcDataResults(curRes)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
    print(allAccuracies, allF1)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3)
trainData = [(X_train[i], np.array([y_train[i]])) for i in range(len(X_train))]
testData = [(X_test[i], np.array([y_test[i]])) for i in range(len(X_test))]
transformedTrainData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in trainData]
transformedTestData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in testData]
for k in range(len(transformedTrainData)):
    if transformedTrainData[k][1][0][0] == 1:
        transformedTrainData[k] = (transformedTrainData[k][0], np.array([[1.0], [0.0], [0.0]]))
    elif transformedTrainData[k][1][0][0] == 2:
        transformedTrainData[k] = (transformedTrainData[k][0], np.array([[0.0], [1.0], [0.0]]))
    else:
        transformedTrainData[k] = (transformedTrainData[k][0], np.array([[0.0], [0.0], [1.0]]))
for k in range(len(transformedTestData)):
    if transformedTestData[k][1][0][0] == 1:
        transformedTestData[k] = (transformedTestData[k][0], np.array([[1.0], [0.0], [0.0]]))
    elif transformedTestData[k][1][0][0] == 2:
        transformedTestData[k] = (transformedTestData[k][0], np.array([[0.0], [1.0], [0.0]]))
    else:
        transformedTestData[k] = (transformedTestData[k][0], np.array([[0.0], [0.0], [1.0]]))
allErorrs = []
nnValues = copy.deepcopy(PREDETERMINED_VALUES[4])
nnValues[0].insert(0, len(transformedTrainData[0][0]))
nnValues[0].append(3)
for q in range(20, len(transformedTrainData) + 1, 20):
    nn = NeuralNetwork(nnValues[0], False)
    nn.regularization_param = nnValues[1]
    nn.train(transformedTrainData[:q], 1000, nnValues[2], math.e ** -8)
    total_cost = 0
    for x, y in transformedTestData:
        predicted = nn.feedforward(x)
        instance_cost = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))
        total_cost += instance_cost
    avg_cost = total_cost / len(transformedTrainData)
    S = sum(np.sum(theta ** 2) for theta in nn.weights)
    S *= (nn.regularization_param / (2 * len(transformedTrainData)))
    avg_cost += S
    allErorrs.append(avg_cost)
trainingNumInst = range(20, len(transformedTrainData) + 1, 20)
plt.plot(trainingNumInst, allErorrs, marker='o', linestyle='-')
plt.xlabel('Number of training instances')
plt.ylabel('Error J of testing data')
plt.title('Error J vs training instances Wine graph')
plt.grid(True)
plt.show()


print("Data of Cancer")
df = pd.read_csv('hw3_cancer.csv', sep='\t')
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X_encoded = np.array(fitData(X))
unClasses, counts = np.unique(Y, return_counts=True)
allFolds = create10Folds(X_encoded, Y, unClasses, counts)
for q in range(6):
    allAccuracies = []
    allPrecisions = []
    allRecalls = []
    allF1 = []
    curRes = []
    nnValues = []
    for n in range(len(allFolds)):
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for m in range(len(allFolds)):
            if n == m:
                curTestMeasure = allFolds[n][0]
                curTestClass = allFolds[n][1]
            else:
                for foldMeasure in allFolds[n][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[n][1]:
                    curTrainClass.append(foldClass)
        trainData = [(curTrainMeasure[i], np.array([curTrainClass[i]])) for i in range(len(curTrainMeasure))]
        testData = [(curTestMeasure[i], np.array([curTestClass[i]])) for i in range(len(curTestMeasure))]
        transformedTrainData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in trainData]
        transformedTestData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in testData]
        if n == 0:
            nnValues = copy.deepcopy(PREDETERMINED_VALUES[q])
            nnValues[0].insert(0, len(transformedTrainData[0][0]))
            nnValues[0].append(1)
        nn = NeuralNetwork(nnValues[0], False)
        nn.regularization_param = nnValues[1]
        nn.train(transformedTrainData, 1000, nnValues[2], math.e ** -8)
        resPredictions = []
        for x, y in transformedTestData:
            resPredictions.append([y[0][0], round(nn.feedforward(x)[0][0])])
        curRes.append(resPredictions)
    print("Now testing the Cancer dataset with the following values: Layers = ", nnValues[0], ", regularization_param = "
          , nnValues[1], ", alpha = ", nnValues[2])
    accuracy, precision, recall, F1 = calcDataResults(curRes)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
    print("Final Accuracy, Precision based avg throughout 10 folds = ", allAccuracies, allF1)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, test_size=0.3)
trainData = [(X_train[i], np.array([y_train[i]])) for i in range(len(X_train))]
testData = [(X_test[i], np.array([y_test[i]])) for i in range(len(X_test))]
transformedTrainData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in trainData]
transformedTestData = [(np.array(arr[0]).reshape(-1, 1), np.array(arr[1]).reshape(-1, 1)) for arr in testData]
allErorrs = []
nnValues = copy.deepcopy(PREDETERMINED_VALUES[3])
nnValues[0].insert(0, len(transformedTrainData[0][0]))
nnValues[0].append(1)
for q in range(70, len(transformedTrainData) + 1, 70):
    nn = NeuralNetwork(nnValues[0], False)
    nn.regularization_param = nnValues[1]
    nn.train(transformedTrainData[:q], 1000, nnValues[2], math.e ** -8)
    total_cost = 0
    for x, y in transformedTestData:
        predicted = nn.feedforward(x)
        instance_cost = np.sum(-y * np.log(predicted) - (1 - y) * np.log(1 - predicted))
        total_cost += instance_cost
    avg_cost = total_cost / len(transformedTrainData)
    S = sum(np.sum(theta ** 2) for theta in nn.weights)
    S *= (nn.regularization_param / (2 * len(transformedTrainData)))
    avg_cost += S
    allErorrs.append(avg_cost)
trainingNumInst = range(70, len(transformedTrainData) + 1, 70)
plt.plot(trainingNumInst, allErorrs, marker='o', linestyle='-')
plt.xlabel('Number of training instances')
plt.ylabel('Error J of testing data')
plt.title('Error J vs training instances cancer graph')
plt.grid(True)
plt.show()


