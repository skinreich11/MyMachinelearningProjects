import csv
import numpy
import numpy as np
import random
from collections import Counter
import pandas as pd
import math
import matplotlib.pyplot as plt

# To run my algorithm, just run the Python file, it should produce all 4 graphs of accuracy, precision, recall,
# and F1 over nTree values for the House data set, Wine data set, and cancer data set, with corresponding labels.
# I chose k = 10 for this assignment, the average of numerical attributes for splitting a node, and
# the minimal_gain criterion, where if 90 percent of an instance data agree on a class, the node stops and makes a
# prediction on the popular class. When I was testing the impact of changing the minimal_gain criterion in EC1/2,
# I changed it to 100%, so there must be complete agreement on the class of a node to make a prediction.
# To see all graphs, close the current window of the current graph to show the next one.


class decisionTreeInst:
    def __init__(self, measures, classes):
        self.measures = measures
        self.classes = classes
        self.classIdx = None
        self.threshold = None
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
    average = None
    numeral = any(isinstance(x, float) or x > 4 for x in attributes)
    if numeral:
        average = sum(attributes) / len(attributes)
        for k in range(len(attributes)):
            if attributes[k] < average:
                res[0].append(classes[k])
            else:
                res[1].append(classes[k])
    else:
        for k in range(len(attributes)):
            res[attributes[k]].append(classes[k])
    return res, average

def splitDoubleArray(attributes, classes, ind, numeral):
    child1 = [[], []]
    child2 = [[], []]
    child3 = [[], []]
    if numeral:
        curAttributes = [row[ind] for row in attributes]
        average = sum(curAttributes) / len(curAttributes)
        for k in range(len(attributes)):
            if attributes[k][ind] < average:
                child1[0].append(attributes[k])
                child1[1].append(classes[k])
            else:
                child2[0].append(attributes[k])
                child2[1].append(classes[k])
    else:
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


def splitClass(measures, classes, M):
    resInfoGain = -1
    resClassInd = None
    resNum = None
    if M > len(measures[0]):
        ind = random.sample(range(len(measures[0])), len(measures[0]))
    else:
        ind = random.sample(range(len(measures[0])), M)
    for j in ind:
        curAttributes = [row[j] for row in measures]
        splitClasses, cond = splitArray(curAttributes, classes)
        curInfoGain = infoGain(splitClasses[0], splitClasses[1], splitClasses[2], classes)
        if curInfoGain > resInfoGain:
            resInfoGain = curInfoGain
            resClassInd = j
            resNum = cond
    return resClassInd, resNum


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

def splitClassGini(measures, classes, M):
    resGini = -1
    resClassInd = None
    resNum = None
    if M > len(measures[0]):
        ind = random.sample(range(len(measures[0])), len(measures[0]))
    else:
        ind = random.sample(range(len(measures[0])), M)
    for j in ind:
        curAttributes = [row[j] for row in measures]
        splitClasses, cond = splitArray(curAttributes, classes)
        curGini = giniCalc(splitClasses[0], splitClasses[1], splitClasses[2], classes)
        if curGini > resGini:
            resGini = curGini
            resClassInd = j
            resNum = cond
    return resClassInd, resNum


def buildDecisionTree(inst, infoGain, major, M):
    if entropy(inst.classes) == 0:
        inst.prediction = inst.classes[0]
        return
    if major:
        mostFreq = Counter(inst.classes).most_common()[0]
        if mostFreq[1] / len(inst.classes) > 0.90:
            inst.prediction = mostFreq[0]
            return
    if infoGain:
        bestClassIndx, keepIdx = splitClass(inst.measures, inst.classes, M)
    else:
        bestClassIndx, keepIdx = splitClassGini(inst.measures, inst.classes, M)
    if bestClassIndx != -1:
        inst.classIdx = bestClassIndx
        child1, child2, child3 = splitDoubleArray(inst.measures, inst.classes, bestClassIndx, keepIdx)
        if keepIdx is not None:
            inst.threshold = keepIdx
            if child1[0]:
                inst.child1 = decisionTreeInst([[row[n] for n in range(len(row))] for row in child1[0]], child1[1])
                buildDecisionTree(inst.child1, infoGain, major, M)
            if child2[0]:
                inst.child2 = decisionTreeInst([[row[n] for n in range(len(row))] for row in child2[0]], child2[1])
                buildDecisionTree(inst.child2, infoGain, major, M)
            if child3[0]:
                inst.child3 = decisionTreeInst([[row[n] for n in range(len(row))] for row in child3[0]], child3[1])
                buildDecisionTree(inst.child3, infoGain, major, M)
        else:
            if child1[0]:
                inst.child1 = decisionTreeInst([[row[n] for n in range(len(row)) if n != bestClassIndx] for row in child1[0]], child1[1])
                buildDecisionTree(inst.child1, infoGain, major, M)
            if child2[0]:
                inst.child2 = decisionTreeInst([[row[n] for n in range(len(row)) if n != bestClassIndx] for row in child2[0]], child2[1])
                buildDecisionTree(inst.child2, infoGain, major, M)
            if child3[0]:
                inst.child3 = decisionTreeInst([[row[n] for n in range(len(row)) if n != bestClassIndx] for row in child3[0]], child3[1])
                buildDecisionTree(inst.child3, infoGain, major, M)
    else:
        uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
        majorityClass = uniqueClasses[np.argmax(counts)]
        inst.prediction = majorityClass

class RandomForest:
    def __init__(self):
        self.trees = []

    def addTree(self, tree):
        self.trees.append(tree)

    def testRandomForest(self, testMeasures, testClasses):
        resPredictions = []
        for a in range(len(testClasses)):
            curPrediction = []
            for tree in self.trees:
                curPrediction.append(predictClass(tree, testMeasures[a], testClasses[a]))
            resPredictions.append([testClasses[a], Counter(curPrediction).most_common(1)[0][0]])
        return resPredictions


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

def bootsrapDataset(measures, classes):
    npMeasures = np.array(measures)
    npClasses = np.array(classes)
    inx = np.random.choice(len(measures), size=len(measures), replace=True)
    return npMeasures[inx], npClasses[inx]

def predictClass(inst, testingData, testingClasse):
    if inst.prediction is not None:
        return inst.prediction
    if inst.threshold is not None:
        if testingData[inst.classIdx] < inst.threshold:
            if inst.child1 is not None:
                return predictClass(inst.child1, testingData, testingClasse)
            else:
                uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
                majorityClass = uniqueClasses[np.argmax(counts)]
                return majorityClass
        else:
            if inst.child2 is not None:
                return predictClass(inst.child2, testingData, testingClasse)
            else:
                uniqueClasses, counts = np.unique(inst.classes, return_counts=True)
                majorityClass = uniqueClasses[np.argmax(counts)]
                return majorityClass
    else:
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
            curPrecision.append(TP/(TP+FP))
            curRecall.append(TP/(TP+FN))
        resPrecision.append(sum(curPrecision) / len(curPrecision))
        resRecall.append(sum(curRecall) / len(curRecall))
    for res in range(len(resPrecision)):
        resF1.append(2*((resPrecision[res]*resRecall[res])/(resPrecision[res]+resRecall[res])))
    print(len(resAccuracy), len(resPrecision), len(resRecall), len(resF1))
    return sum(resAccuracy) / len(resAccuracy), sum(resPrecision) / len(resPrecision), \
           sum(resRecall) / len(resRecall), sum(resF1) / len(resF1)

def plotFunc(y, title, yLabel):
    plt.figure(figsize=(8, 6))
    plt.plot(ntree, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel('nTrees')
    plt.ylabel(yLabel)
    plt.grid(True)
    plt.show()

ntree = [1, 5, 10, 20, 30, 40, 50]

#accuracy, precision, recall, F1 for house of Votes using minimal_gain = 90% InfoGain
data = pd.read_csv("hw3_house_votes_84.csv")
classArray = data.iloc[:, -1].values
measuresArray = data.iloc[:, :16].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, True, True, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "House Accuracy minimal_gain = 90% InfoGain", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "House Precision minimal_gain = 90% InfoGain", "Precision Over 10 Folds")
plotFunc(allRecalls, "House Recall minimal_gain = 90% InfoGain", "Recall Over 10 Folds")
plotFunc(allF1, "House F1 minimal_gain = 90% InfoGain", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for wine using minimal_gain = 90% InfoGain
data = pd.read_csv("hw3_wine.csv", sep='\t')
classArray = data.iloc[:, 0].values
measuresArray = data.iloc[:, 1:].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, True, True, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "Wine Accuracy minimal_gain = 90% InfoGain", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "Wine Precision minimal_gain = 90% InfoGain", "Precision Over 10 Folds")
plotFunc(allRecalls, "Wine Recall minimal_gain = 90% InfoGain", "Recall Over 10 Folds")
plotFunc(allF1, "Wine F1 minimal_gain = 90% InfoGain", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for House of Votes using minimal_gain = 90% Gini
data = pd.read_csv("hw3_house_votes_84.csv")
classArray = data.iloc[:, -1].values
measuresArray = data.iloc[:, :16].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, False, True, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "House Accuracy minimal_gain = 90% Gini", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "House Precision minimal_gain = 90% Gini", "Precision Over 10 Folds")
plotFunc(allRecalls, "House Recall minimal_gain = 90% Gini", "Recall Over 10 Folds")
plotFunc(allF1, "House F1 minimal_gain = 90% Gini", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for wine using minimal_gain = 90% Gini
data = pd.read_csv("hw3_wine.csv", sep='\t')
classArray = data.iloc[:, 0].values
measuresArray = data.iloc[:, 1:].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, False, True, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "Wine Accuracy minimal_gain = 90% Gini", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "Wine Precision minimal_gain = 90% Gini", "Precision Over 10 Folds")
plotFunc(allRecalls, "Wine Recall minimal_gain = 90% Gini", "Recall Over 10 Folds")
plotFunc(allF1, "Wine F1 minimal_gain = 90% Gini", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for house of votes using minimal_gain = 100% Gini
data = pd.read_csv("hw3_house_votes_84.csv")
classArray = data.iloc[:, -1].values
measuresArray = data.iloc[:, :16].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, False, False, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "House Accuracy minimal_gain = 100% Gini", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "House Precision minimal_gain = 100% Gini", "Precision Over 10 Folds")
plotFunc(allRecalls, "House Recall minimal_gain = 100% Gini", "Recall Over 10 Folds")
plotFunc(allF1, "House F1 minimal_gain = 100% Gini", "F1 Over 10 Folds")


#accuracy, precision, recall, F1 for wine using minimal_gain = 100% Gini
data = pd.read_csv("hw3_wine.csv", sep='\t')
classArray = data.iloc[:, 0].values
measuresArray = data.iloc[:, 1:].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, False, False, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "Wine Accuracy minimal_gain = 100% Gini", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "Wine Precision minimal_gain = 100% Gini", "Precision Over 10 Folds")
plotFunc(allRecalls, "Wine Recall minimal_gain = 100% Gini", "Recall Over 10 Folds")
plotFunc(allF1, "Wine F1 minimal_gain = 100% Gini", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for Cancer using minimal_gain = 90% InfoGain
data = pd.read_csv("hw3_cancer.csv", sep='\t')
classArray = data.iloc[:, -1].values
measuresArray = data.iloc[:, :9].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, True, True, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "Cancer Accuracy minimal_gain = 90% InfoGain", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "Cancer Precision minimal_gain = 90% InfoGain", "Precision Over 10 Folds")
plotFunc(allRecalls, "Cancer Recall minimal_gain = 90% InfoGain", "Recall Over 10 Folds")
plotFunc(allF1, "Cancer F1 minimal_gain = 90% InfoGain", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for Cancer using minimal_gain = 90% Gini
data = pd.read_csv("hw3_cancer.csv", sep='\t')
classArray = data.iloc[:, -1].values
measuresArray = data.iloc[:, :9].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, False, True, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "Cancer Accuracy minimal_gain = 90% Gini", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "Cancer Precision minimal_gain = 90% Gini", "Precision Over 10 Folds")
plotFunc(allRecalls, "Cancer Recall minimal_gain = 90% Gini", "Recall Over 10 Folds")
plotFunc(allF1, "Cancer F1 minimal_gain = 90% Gini", "F1 Over 10 Folds")

#accuracy, precision, recall, F1 for Cancer using minimal_gain = 100% Gini
data = pd.read_csv("hw3_cancer.csv", sep='\t')
classArray = data.iloc[:, -1].values
measuresArray = data.iloc[:, :9].values
numberOfAttr = int(math.sqrt(len(measuresArray[0])))
unClasses, counts = numpy.unique(classArray, return_counts=True)
allFolds = create10Folds(measuresArray, classArray, unClasses, counts)
allAccuracies = []
allPrecisions = []
allRecalls = []
allF1 = []
for i in ntree:
    curResults = []
    for j in range(10):
        randFor = RandomForest()
        curTrainMeasure = []
        curTrainClass = []
        curTestMeasure = []
        curTestClass = []
        for l in range(len(allFolds)):
            if j == l:
                curTestMeasure = allFolds[l][0]
                curTestClass = allFolds[l][1]
            else:
                for foldMeasure in allFolds[l][0]:
                    curTrainMeasure.append(foldMeasure)
                for foldClass in allFolds[l][1]:
                    curTrainClass.append(foldClass)
        for m in range(i):
            bootstrapMeasure, bootstrapClass = bootsrapDataset(curTrainMeasure, curTrainClass)
            newTree = decisionTreeInst(bootstrapMeasure, bootstrapClass)
            buildDecisionTree(newTree, False, False, numberOfAttr)
            randFor.addTree(newTree)
        curResults.append(randFor.testRandomForest(curTestMeasure, curTestClass))
    accuracy, precision, recall, F1 = calcDataResults(curResults)
    allAccuracies.append(accuracy)
    allPrecisions.append(precision)
    allRecalls.append(recall)
    allF1.append(F1)
plotFunc(allAccuracies, "Cancer Accuracy minimal_gain = 100% Gini", "Accuracy Over 10 Folds")
plotFunc(allPrecisions, "Cancer Precision minimal_gain = 100% Gini", "Precision Over 10 Folds")
plotFunc(allRecalls, "Cancer Recall minimal_gain = 100% Gini", "Recall Over 10 Folds")
plotFunc(allF1, "Cancer F1 minimal_gain = 100% Gini", "F1 Over 10 Folds")