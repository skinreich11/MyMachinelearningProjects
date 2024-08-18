# To duplicate the results, just run the Python file run.py with a Python interpreter, making sure that numpy is
# downloaded. All the data should be printed out to the console with respective labels, Once the graph is shown close
# the window to continue running the script.

from utils import *
import pprint
import numpy as np
import math
import matplotlib.pyplot as plt

def naive_bayes(pPosTrain, pNegTrain, pPosTest, pNegTest):
	percentage_positive_instances_train = pPosTrain
	percentage_negative_instances_train = pNegTrain

	percentage_positive_instances_test  = pPosTest
	percentage_negative_instances_test  = pNegTest
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	print("Number of positive training instances:", len(pos_train))
	print("Number of negative training instances:", len(neg_train))
	print("Number of positive test instances:", len(pos_test))
	print("Number of negative test instances:", len(neg_test))
	return pos_train, neg_train, pos_test, neg_test, vocab


def calcProb(log, alpha, text):
	numCorPos = 0
	numCorNeg = 0
	numFalsePos = 0
	numFalseNeg = 0
	for i in range(len(testPos)):
		if log:
			curPosProb = math.log(len(trainPosML) / (len(trainPosML) + len(trainNegML)))
			curNegProb = math.log(len(trainNegML) / (len(trainPosML) + len(trainNegML)))
		else:
			curPosProb = (len(trainPosML) / (len(trainPosML) + len(trainNegML)))
			curNegProb = (len(trainNegML) / (len(trainPosML) + len(trainNegML)))
		npTestPos = np.unique(testPos[i])
		for l in range(len(npTestPos)):
			if npTestPos[l] in wordsFreqML:
				if log and alpha != 0:
					curPosProb += math.log((wordsFreqML[npTestPos[l]][0] + alpha) / (numWordsPosML + alpha*len(wordsFreqML)))
					curNegProb += math.log((wordsFreqML[npTestPos[l]][1] + alpha) / (numWordsNegML + alpha*len(wordsFreqML)))
				elif log:
					curPosProb += math.log((wordsFreqML[npTestPos[l]][0] / numWordsPosML) + 10 ** -8)
					curNegProb += math.log((wordsFreqML[npTestPos[l]][1] / numWordsNegML) + 10 ** -8)
				else:
					curPosProb *= ((wordsFreqML[npTestPos[l]][0] + alpha) / (numWordsPosML + alpha*len(wordsFreqML)))
					curNegProb *= ((wordsFreqML[npTestPos[l]][1] + alpha) / (numWordsNegML + alpha*len(wordsFreqML)))
		if curPosProb > curNegProb:
			numCorPos += 1
		elif curNegProb > curPosProb:
			numFalseNeg += 1
		else:
			rand = random.randint(0, 1)
			if rand == 0:
				numCorPos += 1
			else:
				numFalseNeg += 1
	for i in range(len(testNeg)):
		if log:
			curPosProb = math.log(len(trainPosML) / (len(trainPosML) + len(trainNegML)))
			curNegProb = math.log(len(trainNegML) / (len(trainPosML) + len(trainNegML)))
		else:
			curPosProb = (len(trainPosML) / (len(trainPosML) + len(trainNegML)))
			curNegProb = (len(trainNegML) / (len(trainPosML) + len(trainNegML)))
		npTestNeg = np.unique(testNeg[i])
		for l in range(len(npTestNeg)):
			if npTestNeg[l] in wordsFreqML:
				if log and alpha != 0:
					curPosProb += math.log((wordsFreqML[npTestNeg[l]][0] + alpha) / (numWordsPosML + alpha*len(wordsFreqML)))
					curNegProb += math.log((wordsFreqML[npTestNeg[l]][1] + alpha) / (numWordsNegML + alpha*len(wordsFreqML)))
				elif log:
					curPosProb += math.log((wordsFreqML[npTestNeg[l]][0] / numWordsPosML) + 10 ** -8)
					curNegProb += math.log((wordsFreqML[npTestNeg[l]][1] / numWordsNegML) + 10 ** -8)
				else:
					curPosProb *= ((wordsFreqML[npTestNeg[l]][0] + alpha) / (numWordsPosML + alpha*len(wordsFreqML)))
					curNegProb *= ((wordsFreqML[npTestNeg[l]][1] + alpha) / (numWordsNegML + alpha*len(wordsFreqML)))
		if curNegProb > curPosProb:
			numCorNeg += 1
		elif curPosProb > curNegProb:
			numFalsePos += 1
		else:
			rand = random.randint(0, 1)
			if rand == 0:
				numFalsePos += 1
			else:
				numCorNeg += 1
	if text != "":
		print(text + "accuracy =", (numCorPos + numCorNeg) / (len(testPos) + len(testNeg)))
		print(text + "precision =", numCorPos / (numCorPos + numFalsePos))
		print(text + "recall =", numCorPos / (numCorPos + numFalseNeg))
		print(text + "confusion matrix =\n", "\t\t\tpredicted True\tpredicted False\n", "actual True", numCorPos, '\t\t\t', numFalseNeg,
		  	'\n', "actual False", numFalsePos, '\t\t\t', numCorNeg)
	return (numCorPos + numCorNeg) / (len(testPos) + len(testNeg))


def trainML(log, alpha):
	PosML = []
	NegML = []
	for i in range(len(trainPos)):
		if log:
			curPosProb = math.log(len(trainPos) / (len(trainPos) + len(trainNeg)))
			curNegProb = math.log(len(trainNeg) / (len(trainPos) + len(trainNeg)))
		else:
			curPosProb = (len(trainPos) / (len(trainPos) + len(trainNeg)))
			curNegProb = (len(trainNeg) / (len(trainPos) + len(trainNeg)))
		npTrainPos = np.unique(trainPos[i])
		for l in range(len(npTrainPos)):
			if npTrainPos[l] in wordsFreq:
				if log and alpha != 0:
					curPosProb += math.log((wordsFreq[npTrainPos[l]][0] + alpha) / (numWordsPos + alpha*len(wordsFreq)))
					curNegProb += math.log((wordsFreq[npTrainPos[l]][1] + alpha) / (numWordsNeg + alpha*len(wordsFreq)))
				elif log:
					curPosProb += math.log((wordsFreq[npTrainPos[l]][0] / numWordsPos) + 10 ** -8)
					curNegProb += math.log((wordsFreq[npTrainPos[l]][1] / numWordsNeg) + 10 ** -8)
				else:
					curPosProb *= ((wordsFreq[npTrainPos[l]][0] + alpha) / (numWordsPos + alpha*len(wordsFreq)))
					curNegProb *= ((wordsFreq[npTrainPos[l]][1] + alpha) / (numWordsNeg + alpha*len(wordsFreq)))
		if curPosProb > curNegProb:
			PosML.append(trainPos[i])
		elif curNegProb > curPosProb:
			NegML.append(trainPos[i])
		else:
			rand = random.randint(0, 1)
			if rand == 0:
				PosML.append(trainPos[i])
			else:
				NegML.append(trainPos[i])
	for i in range(len(trainNeg)):
		if log:
			curPosProb = math.log(len(trainPos) / (len(trainPos) + len(trainNeg)))
			curNegProb = math.log(len(trainNeg) / (len(trainPos) + len(trainNeg)))
		else:
			curPosProb = (len(trainPos) / (len(trainPos) + len(trainNeg)))
			curNegProb = (len(trainNeg) / (len(trainPos) + len(trainNeg)))
		npTrainNeg = np.unique(trainNeg[i])
		for l in range(len(npTrainNeg)):
			if npTrainNeg[l] in wordsFreq:
				if log and alpha != 0:
					curPosProb += math.log((wordsFreq[npTrainNeg[l]][0] + alpha) / (numWordsPos + alpha*len(wordsFreq)))
					curNegProb += math.log((wordsFreq[npTrainNeg[l]][1] + alpha) / (numWordsNeg + alpha*len(wordsFreq)))
				elif log:
					curPosProb += math.log((wordsFreq[npTrainNeg[l]][0] / numWordsPos) + 10 ** -8)
					curNegProb += math.log((wordsFreq[npTrainNeg[l]][1] / numWordsNeg) + 10 ** -8)
				else:
					curPosProb *= ((wordsFreq[npTrainNeg[l]][0] + alpha) / (numWordsPos + alpha*len(wordsFreq)))
					curNegProb *= ((wordsFreq[npTrainNeg[l]][1] + alpha) / (numWordsNeg + alpha*len(wordsFreq)))
		if curNegProb > curPosProb:
			NegML.append(trainNeg[i])
		elif curPosProb > curNegProb:
			PosML.append(trainNeg[i])
		else:
			rand = random.randint(0, 1)
			if rand == 0:
				PosML.append(trainNeg[i])
			else:
				NegML.append(trainNeg[i])
	return PosML, NegML


# Q1
trainPos, trainNeg, testPos, testNeg, words = naive_bayes(0.2, 0.2, 0.2, 0.2)
wordsFreq = {i: [0, 0] for i in words}
numWordsPos = 0
numWordsNeg = 0
for i in range(len(trainPos)):
	for j in range(len(trainPos[i])):
		numWordsPos += 1
		wordsFreq[trainPos[i][j]][0] += 1
for i in range(len(trainNeg)):
	for j in range(len(trainNeg[i])):
		numWordsNeg += 1
		wordsFreq[trainNeg[i][j]][1] += 1
trainPosML, trainNegML = trainML(False, 0)
wordsFreqML = {i: [0, 0] for i in words}
numWordsPosML = 0
numWordsNegML = 0
for i in range(len(trainPosML)):
	for j in range(len(trainPosML[i])):
		numWordsPosML += 1
		wordsFreqML[trainPosML[i][j]][0] += 1
for i in range(len(trainNegML)):
	for j in range(len(trainNegML[i])):
		numWordsNegML += 1
		wordsFreqML[trainNegML[i][j]][1] += 1
calcProb(False, 0, "Q1 Standard Equation ")
trainPosML, trainNegML = trainML(True, 0)
wordsFreqML = {i: [0, 0] for i in words}
numWordsPosML = 0
numWordsNegML = 0
for i in range(len(trainPosML)):
	for j in range(len(trainPosML[i])):
		numWordsPosML += 1
		wordsFreqML[trainPosML[i][j]][0] += 1
for i in range(len(trainNegML)):
	for j in range(len(trainNegML[i])):
		numWordsNegML += 1
		wordsFreqML[trainNegML[i][j]][1] += 1
calcProb(True, 0, "Q1 Logarithmic Equation ")
# Q2
trainPosML, trainNegML = trainML(True, 1)
wordsFreqML = {i: [0, 0] for i in words}
numWordsPosML = 0
numWordsNegML = 0
for i in range(len(trainPosML)):
	for j in range(len(trainPosML[i])):
		numWordsPosML += 1
		wordsFreqML[trainPosML[i][j]][0] += 1
for i in range(len(trainNegML)):
	for j in range(len(trainNegML[i])):
		numWordsNegML += 1
		wordsFreqML[trainNegML[i][j]][1] += 1
calcProb(True, 1, "Q2 alpha = 1 Logarithmic Equation ")
al = 0.0001
accuracies = []
while al < 1001:
	trainPosML, trainNegML = trainML(True, al)
	wordsFreqML = {i: [0, 0] for i in words}
	numWordsPosML = 0
	numWordsNegML = 0
	for i in range(len(trainPosML)):
		for j in range(len(trainPosML[i])):
			numWordsPosML += 1
			wordsFreqML[trainPosML[i][j]][0] += 1
	for i in range(len(trainNegML)):
		for j in range(len(trainNegML[i])):
			numWordsNegML += 1
			wordsFreqML[trainNegML[i][j]][1] += 1
	accuracies.append(calcProb(True, al, ""))
	al *= 10
xValues = np.logspace(-4, 3, num=len(accuracies))
plt.plot(xValues, accuracies, marker='o', linestyle='-')
plt.xscale('log')
plt.xlabel('alpha')
plt.ylabel('Accuracy')
plt.title('alpha curve with accuracies')
plt.grid(True)
plt.show()
# Q3
trainPos, trainNeg, testPos, testNeg, words = naive_bayes(1, 1, 1, 1)
wordsFreq = {i: [0, 0] for i in words}
numWordsPos = 0
numWordsNeg = 0
for i in range(len(trainPos)):
	for j in range(len(trainPos[i])):
		numWordsPos += 1
		wordsFreq[trainPos[i][j]][0] += 1
for i in range(len(trainNeg)):
	for j in range(len(trainNeg[i])):
		numWordsNeg += 1
		wordsFreq[trainNeg[i][j]][1] += 1
trainPosML, trainNegML = trainML(True, 1)
wordsFreqML = {i: [0, 0] for i in words}
numWordsPosML = 0
numWordsNegML = 0
for i in range(len(trainPosML)):
	for j in range(len(trainPosML[i])):
		numWordsPosML += 1
		wordsFreqML[trainPosML[i][j]][0] += 1
for i in range(len(trainNegML)):
	for j in range(len(trainNegML[i])):
		numWordsNegML += 1
		wordsFreqML[trainNegML[i][j]][1] += 1
calcProb(True, 1, "Q3 alpha = 1 entire dataset Logarithmic equation ")
# Q4
trainPos, trainNeg, testPos, testNeg, words = naive_bayes(0.5, 0.5, 1, 1)
wordsFreq = {i: [0, 0] for i in words}
numWordsPos = 0
numWordsNeg = 0
for i in range(len(trainPos)):
	for j in range(len(trainPos[i])):
		numWordsPos += 1
		wordsFreq[trainPos[i][j]][0] += 1
for i in range(len(trainNeg)):
	for j in range(len(trainNeg[i])):
		numWordsNeg += 1
		wordsFreq[trainNeg[i][j]][1] += 1
trainPosML, trainNegML = trainML(True, 1)
wordsFreqML = {i: [0, 0] for i in words}
numWordsPosML = 0
numWordsNegML = 0
for i in range(len(trainPosML)):
	for j in range(len(trainPosML[i])):
		numWordsPosML += 1
		wordsFreqML[trainPosML[i][j]][0] += 1
for i in range(len(trainNegML)):
	for j in range(len(trainNegML[i])):
		numWordsNegML += 1
		wordsFreqML[trainNegML[i][j]][1] += 1
calcProb(True, 1, "Q4 alpha = 1 50% training set Logarithmic equation ")
# Q6
trainPos, trainNeg, testPos, testNeg, words = naive_bayes(0.1, 0.5, 1, 1)
wordsFreq = {i: [0, 0] for i in words}
numWordsPos = 0
numWordsNeg = 0
for i in range(len(trainPos)):
	for j in range(len(trainPos[i])):
		numWordsPos += 1
		wordsFreq[trainPos[i][j]][0] += 1
for i in range(len(trainNeg)):
	for j in range(len(trainNeg[i])):
		numWordsNeg += 1
		wordsFreq[trainNeg[i][j]][1] += 1
trainPosML, trainNegML = trainML(True, 1)
wordsFreqML = {i: [0, 0] for i in words}
numWordsPosML = 0
numWordsNegML = 0
for i in range(len(trainPosML)):
	for j in range(len(trainPosML[i])):
		numWordsPosML += 1
		wordsFreqML[trainPosML[i][j]][0] += 1
for i in range(len(trainNegML)):
	for j in range(len(trainNegML[i])):
		numWordsNegML += 1
		wordsFreqML[trainNegML[i][j]][1] += 1
calcProb(True, 1, "Q6 alpha = 1 training 10% positive 50% negative Logarithmic Equation ")


