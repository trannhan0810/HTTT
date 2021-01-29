import numpy as np 
np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})

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
    return matrix/np.sum(matrix, axis=0)

def getWeightVector(matrix):
    w = np.sum(matrix, axis=1)
    return normalize(w)

def getConsistancyMeasure(matrix, w):
    return matrix.dot(w)/w

RI = [0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.46, 1.49]
def getCR(CM, n):
    if n>10:print("n must <= 10");return 0;
    CI = (np.max(CM) - n)/(n-1)
    CR = CI/RI[n]
    return CR

def calculate(matrix):
    filled = fill(matrix)
    normalized = normalize(filled)
    w = getWeightVector(normalized)
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
    alternatives = [ salary, lifeQuality, interest, nearness ]
    n = []
    o = []
    for i in range(0, len(alternatives)):
        priorities, inconsistency = calculate(alternatives[i])
        n.append(priorities)
        o.append(inconsistency)
    n = np.array(n)
    criterias = np.array([[1, 5, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    priorities, inconsistency = calculate(criterias)
    priorities = np.array(priorities)
    x = priorities.dot(n)

    print("==================")
    print("Inconsistency: ")
    for i in range(0, len(criterias)):
        print(criteria_name[i]+ ": "+ "{0:0.3f}".format(o[i]))
    print("criteria: "+ str(inconsistency))
    print("==================")
    for i in range(0, len(x)):
        print(jobs[i] + ": ", end="")
        print("{0:0.3f}".format(x[i])) 