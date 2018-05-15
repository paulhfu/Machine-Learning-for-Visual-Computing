import h5py
import matplotlib.pyplot as plt
import numpy as np
from random import randint


def initUnaryProbabilities():
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            predictions[i][j] = randint(1, classes + 1)


def createPairProbMatrix(pairMatrix):

    return pairMatrix


def absorbUnaryPot():
    unaryOnes = []
    onesUnary = []
    for i in range(classes):
        unaryOnes.append(np.multiply(predictions[i][:], np.ones((classes, 1))).transpose())
        onesUnary.append(np.multiply(np.ones((classes, 1)), predictions[i][:]))
        
    potential = []
    for j in range(classes - 1):
        if(j != classes - 2):
            potential.append(pairProp + unaryOnes[j] + onesUnary[j + 1])
        else:
            potential.append(pairProp + onesUnary[j + 1])

    return np.array(potential)


def updatePotentials(index):
    if len(minimas) == 0:
        return 0
    else:
        for i in range(classes):
            for j in range(classes):
                potentials[index][j][i] = potentials[index][j][i] + minimas[i - classes]

    return 1


def calculateMinima():
    index = (int)(len(minimas) / classes)
    updatePotentials(index)

    for i in range(classes):
        minimas.append(np.amin(potentials[index][:, i]))

    minIndex = np.argmin(minimas[-classes:])
    minValue = minimas[-classes:][minIndex]

    if index == 0:
        shortestPath.append(minIndex)

    count = 0
    for x in potentials[index][:, minIndex]:
        if x == minValue:
            shortestPath.append(count)
        count += 1


def calculateShortestPath():

    for i in range(classes - 1):
        calculateMinima()


if __name__ == '__main__':
    classes = 3

    # Example 1
    predictions = np.zeros((classes, classes))
    shortestPath = []
    minimas = []
    costs = []

    initUnaryProbabilities()

    pairProp = createPairProbMatrix(np.matrix([[0,   1,   5], 
                                               [100, 0,   1],
                                               [100, 100, 0]]))
    potentials = absorbUnaryPot()

    calculateShortestPath()
    print("Shortest Path Example 1: ", shortestPath)


    # Example 2
    predictions = np.zeros((classes, classes))
    shortestPath = []
    minimas = []
    costs = []

    initUnaryProbabilities()

    pairProp = createPairProbMatrix(np.matrix([[0,   9,   5], 
                                               [5,   0,   1],
                                               [100, 2,   0]]))
    potentials = absorbUnaryPot()

    calculateShortestPath()
    print("Shortest Path Example 2: ", shortestPath)