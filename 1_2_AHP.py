import numpy as np 
np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})

def fill(matrix):
    length = len(matrix) 
    #Set polygon = 1
    for i in range(0, length): matrix[i,i] = 1
    #If row 0 contain 0 raise a error
    if any(cell == 0 for cell in matrix[0] ): raise "missing number in row 0"
    for i in range(0, length):
        for j in range(0, length):
            if matrix[i,j] == 0:
                #If cell[j,i] have value, use inverse number of cell[j,i]
                if matrix[j,i] != 0: matrix[i,j] = 1/matrix[j,i]
                 #Else use the chain rule: cell[0,j]=cell[0,i]*cell[i,j]
                else: matrix[i,j] = matrix[0,j]/matrix[0,i]
    return matrix

def normalize(comparisonMatrix):
    #Divide each column by it's sum
    matrix = comparisonMatrix.copy()
    return matrix/np.sum(matrix, axis=0)

def getWeightVector(matrix):
    #get size n of matrix
    n = len(matrix[0])
    #Calculste sum each row then divide it by n
    w = np.sum(matrix, axis=1)/n #=> output is a array with n element
    #Reshape it to a matrix n row x 1 col
    return np.reshape(w, (-1, 1))

def getConsistancyMeasure(matrix, w):
    return matrix.dot(w)/w

RI = [0, 0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.46, 1.49]
def getCR(CM, n):
    if n>10: raise "n must <= 10";
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

def AHP(criteria_matrix, list_of_alternatives_matrix):
    #Get weight vector from criteria_comparision_matrix
    criteria_w, criteria_inconsistency = calculate(criteria_matrix)

    list_alter_w, list_alter_inconsistency = [],[]
    #for each alternative_comparison_matrix, get weight vector and combine them into a matrix
    for alternatives_matrix in list_of_alternatives_matrix:
        alter_w, alter_inconsistency = calculate(alternatives_matrix)
        list_alter_w.append(alter_w)
        list_alter_inconsistency.append(alter_inconsistency)

    criteria_alter = np.hstack(list_alter_w)        #combine weight vector of alternative into a matrix
    result = np.dot(criteria_alter, criteria_w)     #result shape is (n,1)
    result = np.reshape(result, (-1))               #change shape to (n)

    return result, criteria_inconsistency, list_alter_inconsistency

#===========================EXAMPLE====================================
if __name__ == '__main__':
    criteria_name   = [ "salary", "lifeQuality", "interest", "nearness" ]
    criterias = np.array([[1, 5, 2, 4], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=float)

    job_names       = ["jobA", "jobB", "jobC"]
    job_salary      = np.array([[0, 2,   4   ],  [0, 0, 2], [0, 0,  0]], dtype=float)
    job_lifeQuality = np.array([[0, 0.33,0.5 ],  [3, 0, 2], [0, 0,  0]], dtype=float)
    job_interest    = np.array([[1, 0.5, 0.33],  [2, 0, 2], [3, 0.5,0]], dtype=float)
    job_nearness    = np.array([[1, 0.25,1   ],  [0, 0, 4], [0, 0,  0]], dtype=float)
    alternatives = [ job_salary, job_lifeQuality, job_interest, job_nearness ]

    result, criteria_inconsistency, list_alter_inconsistency = AHP(criterias, alternatives)

    print("\n==================")
    print("Comparision matrix: ")
    print("criterias: ")
    print(criterias)
    for i in range(0, len(criterias)):
        print("job_" + criteria_name[i] + ": ")
        print(alternatives[i])


    print("\n==================")
    print("Inconsistency: ")
    for i in range(0, len(criterias)):
        print(criteria_name[i]+ ": "+ "{0:0.3f}".format(list_alter_inconsistency[i]))
    print("criterias: "+ str(criteria_inconsistency))

    print("\n==================")
    print("Point for each job")
    for i in range(0, len(result)):
        print(job_names[i] + ": ", end="")
        print("{0:0.3f}".format(result[i])) 
