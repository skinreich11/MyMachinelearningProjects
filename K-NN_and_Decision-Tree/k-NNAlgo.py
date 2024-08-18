# To duplicate the result, just run the Python file k-NNAlgo.py with a Python interpreter making sure that sklearn is
# downloaded on your local machine. Once a data graph is shown, close the graph window to show the next graph. It will
# show all three graphs.
import numpy as np
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv("iris.csv")
measureArray = data.iloc[:, :4].to_numpy()
classArray = data.iloc[:, -1].to_numpy()
trainAccuracy = []
testAccuracy = []
testAccuracyNoNorm = []


def fitData(array):
    maxNum = np.max(array, axis=0)
    minNum = np.min(array, axis=0)
    result = np.zeros_like(array)
    for m in range(len(array)):
        for n in range(len(array[m])):
            result[m][n] = (array[m][n] - minNum[n]) / (maxNum[n] - minNum[n])
    return result


def findDist(point, array):
    result = []
    for m in range(len(array)):
        curDist = 0
        for n in range(len(point)):
            curDist += (point[n] - array[m][n]) ** 2
        result.append(math.sqrt(curDist))
    return result


for i in range(1, 52, 2):
    for j in range(20):
        measureTrain, measureTest, classTrain, classTest = train_test_split(measureArray, classArray, train_size=0.8)
        measureTrain = np.array(fitData(measureTrain))
        measureTest = np.array(fitData(measureTest))
        counterTrain = 0
        counterTest = 0
        kNNMeasure = []
        kNNClass = []
        for pointIdx, point in enumerate(measureTrain):
            eucDist = findDist(point, measureTrain)
            sortedDist = np.argsort(eucDist)
            closePointsClass = [classTrain[k] for k in sortedDist[:i]]
            classCounts = Counter(closePointsClass)
            majorityClassOccur = classCounts.most_common()[0][1]
            majorityClasses = [element for element, count in classCounts.items() if count == majorityClassOccur]
            designation = random.choice(majorityClasses)
            kNNMeasure.append(point)
            kNNClass.append(designation)
        for pointIdx, point in enumerate(measureTrain):
            eucDist = findDist(point, kNNMeasure)
            sortedDist = np.argsort(eucDist)
            closePointsClass = [kNNClass[k] for k in sortedDist[:i]]
            classCounts = Counter(closePointsClass)
            majorityClassOccur = classCounts.most_common()[0][1]
            majorityClasses = [element for element, count in classCounts.items() if count == majorityClassOccur]
            designation = random.choice(majorityClasses)
            if classTrain[pointIdx] == designation:
                counterTrain += 1
        if j == 0:
            trainAccuracy.append([counterTrain / len(measureTrain)])
        else:
            trainAccuracy[-1].append(counterTrain / len(measureTrain))
        for pointIdx, point in enumerate(measureTest):
            eucDist = findDist(point, kNNMeasure)
            sortedDist = np.argsort(eucDist)
            closePointsClass = [kNNClass[k] for k in sortedDist[:i]]
            classCounts = Counter(closePointsClass)
            majorityClassOccur = classCounts.most_common()[0][1]
            majorityClasses = [element for element, count in classCounts.items() if count == majorityClassOccur]
            designation = random.choice(majorityClasses)
            if classTest[pointIdx] == designation:
                counterTest += 1
        if j == 0:
            testAccuracy.append([counterTest / len(measureTest)])
        else:
            testAccuracy[-1].append(counterTest / len(measureTest))
x = np.arange(1, 52, 2)
npTrainAccuracy = np.array(trainAccuracy)
npTestAccuracy = np.array(testAccuracy)
for i in range(npTrainAccuracy.shape[0]):
    y = npTrainAccuracy[i]
    average = np.mean(y)
    std_dev = np.std(y)
    plt.errorbar(x[i], average, yerr=std_dev, fmt='o', label=f'Array {i + 1}')
    if i < npTrainAccuracy.shape[0] - 1:
        plt.plot([x[i], x[i + 1]], [average, np.mean(npTrainAccuracy[i + 1])], color='gray',
                 linestyle='--')
plt.xlabel('K-Value')
plt.ylabel('Accuracy Over Training Data')
plt.title('Training Data Graph K-NN')
plt.show()
for i in range(npTestAccuracy.shape[0]):
    y = npTestAccuracy[i]
    average = np.mean(y)
    std_dev = np.std(y)
    plt.errorbar(x[i], average, yerr=std_dev, fmt='o', label=f'Array {i + 1}')
    if i < npTestAccuracy.shape[0] - 1:
        plt.plot([x[i], x[i + 1]], [average, np.mean(npTestAccuracy[i + 1])], color='gray',
                 linestyle='--')
plt.xlabel('K-Value')
plt.ylabel('Accuracy Over Testing Data')
plt.title('Testing Data Graph k-NN')
plt.show()

for i in range(1, 52, 2):
    for j in range(20):
        measureTrain, measureTest, classTrain, classTest = train_test_split(measureArray, classArray, train_size=0.8)
        counterTest = 0
        kNNMeasure = []
        kNNClass = []
        for pointIdx, point in enumerate(measureTrain):
            eucDist = findDist(point, measureTrain)
            sortedDist = np.argsort(eucDist)
            closePointsClass = [classTrain[k] for k in sortedDist[:i]]
            classCounts = Counter(closePointsClass)
            majorityClassOccur = classCounts.most_common()[0][1]
            majorityClasses = [element for element, count in classCounts.items() if count == majorityClassOccur]
            designation = random.choice(majorityClasses)
            kNNMeasure.append(point)
            kNNClass.append(designation)
        for pointIdx, point in enumerate(measureTest):
            eucDist = findDist(point, kNNMeasure)
            sortedDist = np.argsort(eucDist)
            closePointsClass = [kNNClass[k] for k in sortedDist[:i]]
            classCounts = Counter(closePointsClass)
            majorityClassOccur = classCounts.most_common()[0][1]
            majorityClasses = [element for element, count in classCounts.items() if count == majorityClassOccur]
            designation = random.choice(majorityClasses)
            if classTest[pointIdx] == designation:
                counterTest += 1
        if j == 0:
            testAccuracyNoNorm.append([counterTest / len(measureTest)])
        else:
            testAccuracyNoNorm[-1].append(counterTest / len(measureTest))
npTestAccuracyNoNorm = np.array(testAccuracyNoNorm)
for i in range(npTestAccuracyNoNorm.shape[0]):
    y = npTestAccuracyNoNorm[i]
    average = np.mean(y)
    std_dev = np.std(y)
    plt.errorbar(x[i], average, yerr=std_dev, fmt='o', label=f'Array {i + 1}')
    if i < npTestAccuracyNoNorm.shape[0] - 1:
        plt.plot([x[i], x[i + 1]], [average, np.mean(npTestAccuracyNoNorm[i + 1])], color='gray',
                 linestyle='--')
plt.xlabel('K-Value')
plt.ylabel('Accuracy Over Testing Data')
plt.title('Testing Data Graph k-NN No Normalization')
plt.show()
