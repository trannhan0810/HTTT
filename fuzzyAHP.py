import numpy as np 
from AHP import fill, normalize, getW


def fuzzification(listComparisonMatrix):
    for i in range(0, len(listComparisonMatrix)):
        listComparisonMatrix[i] = fill(listComparisonMatrix[i])
        #print(listComparisonMatrix[i])
  
    L = np.min(listComparisonMatrix, 0)
    M = np.average(listComparisonMatrix, 0)
    U = np.max(listComparisonMatrix, 0)
    return np.dstack((L, M, U))

def printTensor(matrix):
    h, w, d = matrix.shape
    for i in range(0, h):
        for j in range(0, w):
            print(matrix[i][j], end=", ")
        print("")

def fuzz(x):
    l = x-2 if (x-2)>=1 else 1
    u = x+2 if (x+2)<=9 else 9
    return np.array([l, x, u])

def aggregateIndividualPreference(listPrefernce):
    listPrefernce = fuzz(listPrefernce)

listComparisonMatrix = [
    np.array([[0,4,3,2], [0,0,0.8,0.7], [0,0,0,0.6],[0,0,0,0]], dtype=float),
    np.array([[0,4.2,3,1.5], [0,0,0.85,0.7], [0,0,0,0.65],[0,0,0,0]], dtype=float),
    np.array([[0,3.9,2.8,2.3], [0,0,0.7,0.7], [0,0,0,0.65],[0,0,0,0]], dtype=float),
]

prefernce1 = np.array([[2, 1, 3, 4], [4, 3, 1, 2], [3, 2, 2, 2]])
prefernce2 = np.array([[3, 3, 1, 3], [4, 1, 2, 3], [4, 4, 1, 1]])
prefernce3 = np.array([[2, 2, 1, 5], [2, 3, 2, 3], [3, 3, 3, 1]])
listPreference = np.array([prefernce1, prefernce2, prefernce3])

print(fuzz(prefernce1))
#==================================================================
fuzzied = fuzzification(listComparisonMatrix)
fuzzied_normalized = normalize(fuzzied)
w = getW(fuzzied_normalized)


