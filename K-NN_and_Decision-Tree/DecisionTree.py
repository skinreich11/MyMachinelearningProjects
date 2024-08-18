# To duplicate the result, just run the Python file DecisionTree.py with a Python interpreter making sure that
# sklearn is downloaded on your local machine. Once a data graph is shown, close the graph window to show the next
# graph. It will show all 6 graphs, as well as print to console the mean and standard deviation of the
# corresponding graph. The majority extra credit question was run with the infoGain calculations.
import numpy as np
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("house_votes_84.csv")
measureArray = data.iloc[:, :16].to_numpy()
classArray = data.iloc[:, -1].to_numpy()


class decisionTreeInst:
    def __init__(self, measures, classes):
        self.measures = measures
        self.classes = classes
        self.classIdx = None
        self.child1 = None
        self.child2 = None
        self.child3 = None
        self.prediction = None


def infoGain(child1Classes, child2Classes, child3Classes, parentClasses):
    parentEntropy = entropy(parentClasses)
    child1Entropy = entropy(child1Classes)
    child2Entropy = entropy(child2Classes)
    child3Entropy = entropy(child3Classes)
    return parentEntropy - (child1Entropy * (len(child1Classes) / len(parentClasses)) +
                            child2Entropy * (len(child2Classes) / len(parentClasses)) +
                            child3Entropy * (len(child3Classes) / len(parentClasses)))


def entropy(classes):
    _, numInst = np.unique(classes, return_counts=True)
    if len(numInst) == 1:
        return 0
    result = 0
    for inst in numInst:
        result -= (inst / len(classes)) * np.log2(inst / len(classes))
    return result

def splitArray(attributes, classes):
    res = [[], [], []]
    for k in range(len(attributes)):
        res[attributes[k]].append(classes[k])
    return res

def splitDoubleArray(attributes, classes, ind):
    child1 = [[], []]
    child2 = [[], []]
    child3 = [[], []]
    for l in range(len(attributes)):
        if attributes[l][ind] == 0:
            child1[0].append(attributes[l])
            child1[1].append(classes[l])
        if attributes[l][ind] == 1:
            child2[0].append(attributes[l])
            child2[1].append(classes[l])
        if attributes[l][ind] == 2:
            child3[0].append(attributes[l])
            child3[1].append(classes[l])
    return child1, child2, child3


def splitClass(measures, classes):
    resInfoGain = -1
    resClassInd = None
    for j in range(len(measures[0])):
        curAttributes = [row[j] for row in measures]
        splitClasses = splitArray(curAttributes, classes)
        curInfoGain = infoGain(splitClasses[0], splitClasses[1], splitClasses[2], classes)
        if curInfoGain > resInfoGain:
            resInfoGain = curInfoGain
            resClassInd = j
    return resClassInd


def gini(classes):
    _, numInst = np.unique(classes, return_counts=True)
    result = 0
    if len(numInst) == 1:
        return 0
    for inst in numInst:
        result += (inst / len(classes)) ** 2
    return 1 - result


def giniCalc(child1Classes, child2Classes, child3Classes, parentClasses):
    parentEntropy = gini(parentClasses)
    child1Entropy = gini(child1Classes)
    child2Entropy = gini(child2Classes)
    child3Entropy = gini(child3Classes)
    return parentEntropy - (child1Entropy * (len(child1Classes) / len(parentClasses)) +
                            child2Entropy * (len(child2Classes) / len(parentClasses)) +
                            child3Entropy * (len(child3Classes) / len(parentClasses)))

def splitClassGini(measures, classes):
    resGini = -1
    resClassInd = None
    for j in range(len(measures[0])):
        curAttributes = [row[j] for row in measures]
        splitClasses = splitArray(curAttributes, classes)
        curGini = giniCalc(splitClasses[0], splitClasses[1], splitClasses[2], classes)
        if curGini > resGini:
            resGini = curGini
            resClassInd = j
    return resClassInd


def buildDecisionTree(inst, infoGain, major):
    if entropy(inst.classes) == 0:
        inst.prediction = inst.classes[0]
        return
    if major:
        mostFreq = Counter(inst.classes).most_common()[0]
        if mostFreq[1] / len(inst.classes) > 0.85:
            inst.prediction = mostFreq[0]
            return
    if infoGain:
        bestClassIndx = splitClass(inst.measures, inst.classes)
    else:
        bestClassIndx = splitClassGini(inst.measures, inst.classes)
    if bestClassIndx != -1:
        inst.classIdx = bestClassIndx
        child1, child2, child3 = splitDoubleArray(inst.measures, inst.classes, bestClassIndx)
        if child1[0]:
            inst.child1 = decisionTreeInst([[row[n] for n in range(len(row)) if n != bestClassIndx] for row in child1[0]], child1[1])
            buildDecisionTree(inst.child1, infoGain, major)
        if child2[0]:
            inst.child2 = decisionTreeInst([[row[n] for n in range(len(row)) if n != bestClassIndx] for row in child2[0]], child2[1])
            buildDecisionTree(inst.child2, infoGain, major)
        if child3[0]:
            inst.child3 = decisionTreeInst([[row[n] for n in range(len(row)) if n != bestClassIndx] for row in child3[0]], child3[1])
            buildDecisionTree(inst.child3, infoGain, major)
    else:
        uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
        majorityClass = uniqueClasses[np.argmax(counts)]
        inst.prediction = majorityClass

def predictClass(inst, testingData, testingClasse):
    if inst.prediction is not None:
        if inst.prediction == testingClasse:
            return 1
        else:
            return 0

    if testingData[inst.classIdx] == 0:
        if inst.child1 is not None:
            testingData = np.delete(testingData, inst.classIdx)
            return predictClass(inst.child1, testingData, testingClasse)
        else:
            uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
            majorityClass = uniqueClasses[np.argmax(counts)]
            if majorityClass == testingClasse:
                return 1
            else:
                return 0
    if testingData[inst.classIdx] == 1:
        if inst.child2 is not None:
            testingData = np.delete(testingData, inst.classIdx)
            return predictClass(inst.child2, testingData, testingClasse)
        else:
            uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
            majorityClass = uniqueClasses[np.argmax(counts)]
            if majorityClass == testingClasse:
                return 1
            else:
                return 0
    if testingData[inst.classIdx] == 2:
        if inst.child3 is not None:
            testingData = np.delete(testingData, inst.classIdx)
            return predictClass(inst.child3, testingData, testingClasse)
        else:
            uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
            majorityClass = uniqueClasses[np.argmax(counts)]
            if majorityClass == testingClasse:
                return 1
            else:
                return 0


counterTrain = []
counterTest = []
for i in range(100):
    measureTrain, measureTest, classTrain, classTest = train_test_split(measureArray, classArray, train_size=0.8)
    newTree = decisionTreeInst(measureTrain, classTrain)
    buildDecisionTree(newTree, True, False)
    curCounterTrain = 0
    curCounterTest = 0
    for q in range(len(measureTrain)):
        curCounterTrain += predictClass(newTree, measureTrain[q], classTrain[q])
    for v in range(len(measureTest)):
        curCounterTest += predictClass(newTree, measureTest[v], classTest[v])
    counterTrain.append(curCounterTrain / len(measureTrain))
    counterTest.append(curCounterTest / len(measureTest))
print(np.mean(np.array(counterTrain)))
print(np.std(np.array(counterTrain)))
print(np.mean(np.array(counterTest)))
print(np.std(np.array(counterTest)))

plt.hist(counterTrain, edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency')
plt.title('Training Data Graph Decision Tree infoGain')
plt.show()

plt.hist(counterTest, edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency')
plt.title('Testing Data Graph Decision Tree infoGain')
plt.show()

counterTrain = []
counterTest = []
for i in range(100):
    measureTrain, measureTest, classTrain, classTest = train_test_split(measureArray, classArray, train_size=0.8)
    newTree = decisionTreeInst(measureTrain, classTrain)
    buildDecisionTree(newTree, False, False)
    curCounterTrain = 0
    curCounterTest = 0
    for q in range(len(measureTrain)):
        curCounterTrain += predictClass(newTree, measureTrain[q], classTrain[q])
    for v in range(len(measureTest)):
        curCounterTest += predictClass(newTree, measureTest[v], classTest[v])
    counterTrain.append(curCounterTrain / len(measureTrain))
    counterTest.append(curCounterTest / len(measureTest))

print(np.mean(np.array(counterTrain)))
print(np.std(np.array(counterTrain)))
print(np.mean(np.array(counterTest)))
print(np.std(np.array(counterTest)))

plt.hist(counterTrain, edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency')
plt.title('Training Data Graph Decision Tree Gini')
plt.show()

plt.hist(counterTest, edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency')
plt.title('Testing Data Graph Decision Tree Gini')
plt.show()

counterTrain = []
counterTest = []
for i in range(100):
    measureTrain, measureTest, classTrain, classTest = train_test_split(measureArray, classArray, train_size=0.8)
    newTree = decisionTreeInst(measureTrain, classTrain)
    buildDecisionTree(newTree, True, True)
    curCounterTrain = 0
    curCounterTest = 0
    for q in range(len(measureTrain)):
        curCounterTrain += predictClass(newTree, measureTrain[q], classTrain[q])
    for v in range(len(measureTest)):
        curCounterTest += predictClass(newTree, measureTest[v], classTest[v])
    counterTrain.append(curCounterTrain / len(measureTrain))
    counterTest.append(curCounterTest / len(measureTest))

print(np.mean(np.array(counterTrain)))
print(np.std(np.array(counterTrain)))
print(np.mean(np.array(counterTest)))
print(np.std(np.array(counterTest)))

plt.hist(counterTrain, edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency')
plt.title('Training Data Graph Decision Tree Majority85')
plt.show()

plt.hist(counterTest, edgecolor='black')
plt.xlabel('Accuracy')
plt.ylabel('Accuracy Frequency')
plt.title('Testing Data Graph Decision Tree Majority85')
plt.show()

