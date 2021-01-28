import numpy as np 
np.set_printoptions(precision=3)

def fill(matrix):
    m_len = len(matrix) 
    for i in range(0, m_len):
        matrix[i,i] = 1
    for i in range(0, m_len):
        if matrix[0,i] == 0:
            print("missing number in row 0") 
            return [] 
    for i in range(0, m_len):
        for j in range(0, m_len):
            if matrix[i,j] != 0:
                continue
            else:
                if matrix[j,i] != 0:
                    matrix[i,j] = 1/matrix[j,i]
                else:
                    matrix[i,j] = matrix[0,j]/matrix[0,i]
    return matrix

def normalize(comparisonMatrix):
    matrix = comparisonMatrix.copy()
    length = len(matrix)
    for i in range(0, length):
        sumColumn = np.sum(matrix[:,i], axis=0)
        for j in range(0, length):
            matrix[j][i] /= sumColumn
    return matrix

def getW(matrix):
    result = []
    for i in range(0, len(matrix)):
        result.append(np.average(matrix[i], axis=0))
    return np.array(result)

def getConsistancyMeasure(matrix, w):
    W = matrix.dot(w)
    for i in range(0, len(matrix)):
        W[i] = W[i]/w[i]
    return W

def getCR(CM, n):
    if n>10:
        print("n must <= 10")
        return 0
    RI = [0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.46, 1.49]
    CI = (np.max(CM) - n)/(n-1)
    CR = CI/RI[n]
    return CR

def calculate(matrix):
    filled = fill(matrix)
    normalized = normalize(filled)
    w = getW(normalized)
    CM = getConsistancyMeasure(filled, w)
    CR= getCR(CM, 3)
    return w, CR


#=========================================================================
if __name__ == '__main__':
    criteria_name   = [ "salary", "lifeQuality", "interest", "nearness" ]
    jobs            = ["jobA", "jobB", "jobC"]

    salary      = np.array([[0, 2,   4   ],  [0, 0, 2], [0, 0,  0]], dtype=float)
    lifeQuality = np.array([[0, 0.33,0.5 ],  [3, 0, 2], [0, 0,  0]], dtype=float)
    interest    = np.array([[1, 0.5, 0.33],  [2, 0, 2], [3, 0.5,0]], dtype=float)
    nearness    = np.array([[1, 0.25,1   ],  [0, 0, 4], [0, 0,  0]], dtype=float)
    m = [ salary, lifeQuality, interest, nearness ]
    n = []
    o = []
    for i in range(0, len(m)):
        priorities, inconsistency = calculate(m[i])
        n.append(priorities)
        o.append(inconsistency)
    n = np.array(n)
    criteria = np.array([[1, 5, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    priorities, inconsistency = calculate(criteria)
    priorities = np.array(priorities)
    print(priorities.shape)
    print(n.shape)
    x = priorities.dot(n)

    print("==================")
    print("Inconsistency: ")
    for i in range(0, len(m)):
        print(criteria_name[i]+ ": "+ str(o[i]))
    print("criteria: "+ str(inconsistency))
    print("==================")
    for i in range(0, len(x)):
        print(jobs[i] + ": " + str(x[i]) )