import numpy as np 

np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})


def fill(matrix):
    m_len = len(matrix) 
    for i in range(0, m_len):
        matrix[i,i] = 1
    for i in range(0, m_len):
        if matrix[0,i] == 0:
            raise "missing number in row 0"
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

def show(matrix, name="---------"):
    print("---"+name+"----")
    if(matrix.ndim >4): print("Error, can't show matrix with high than 4 dimension"); return;
    if(matrix.ndim ==4):
        for i in range(0,3): show(matrix[i]); 
        return
    if(matrix.ndim >=3):
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                print(matrix[i][j], end=", ")
            print("")
    else:
        print(matrix)

def fillAllComparisonMatrix(matrixs):
    print(matrixs.shape)
    for i in range(0, len(matrixs)):
        matrixs[i] = fill(matrixs[i])
    return matrixs

def aggegate(listComparisonMatrix):
    L = np.min(listComparisonMatrix, axis = 0)
    M = np.average(listComparisonMatrix, axis = 0)
    U = np.max(listComparisonMatrix, axis = 0)
    return np.dstack((L, M, U))

def fuzzyfication(M):
    s = (M.shape)
    L = np.maximum(M-2, np.full(s, 1))
    U = np.minimum(M+2, np.full(s, 9))
    return np.dstack((L, M, U))

def defuzzyfication(fuzzyMatrix, alpha = 0.5, beta = 0.5):
    alpha_l = fuzzyMatrix[:,:,0] + alpha*(fuzzyMatrix[:,:,1] - fuzzyMatrix[:,:,0])
    alpha_r = fuzzyMatrix[:,:,2] - alpha*(fuzzyMatrix[:,:,2] - fuzzyMatrix[:,:,1])
    alpha_matrix = np.dstack((alpha_l, alpha_r))
    show(alpha_matrix, "alpha")

    beta_matrix = np.add(np.multiply(alpha_matrix[:,:,0],beta),np.multiply(alpha_matrix[:,:,1],(1-beta)))
    #show(beta_matrix, "beta")
    
    return beta_matrix

def reorder(matrix):
    if(matrix.ndim != 3): print("ERROR"); return;
    L = np.min([matrix[:,:,0], matrix[:,:,1], matrix[:,:,2]], axis = 0)
    U = np.max([matrix[:,:,0], matrix[:,:,1], matrix[:,:,2]], axis = 0)
    M = matrix[:,:,0] + matrix[:,:,1] + matrix[:,:,2] - L - U
    return np.dstack((L, M, U))

#==================================================================
D1 = np.array([[0.0, 4.0, 3.0, 2.0], 
                [0.0, 0.0, 0.8, 0.7], 
                [0.0, 0.0, 0.0, 0.60], 
                [0, 0, 0, 0]], dtype=float)

D2 = np.array([[0.0, 4.2, 3.0, 1.5], 
                [0.0, 0.0, 0.85,0.7], 
                [0.0, 0.0, 0.0, 0.65], 
                [0, 0, 0, 0]], dtype=float)

D3 = np.array([[0.0, 3.9, 2.8, 2.3], 
                [0.0, 0.0, 0.7, 0.7], 
                [0.0, 0.0, 0.0, 0.65], 
                [0, 0, 0, 0]], dtype=float)
listComparisonMatrix = np.array([D1, D2, D3])

alternative1 = np.array([[2, 1, 3, 4], 
                        [4, 3, 1, 2], 
                        [3, 2, 2, 2]])
alternative2 = np.array([[3, 3, 1, 3], 
                        [4, 1, 2, 3], 
                        [4, 4, 1, 1]])
alternative3 = np.array([[2, 2, 1, 5], 
                        [2, 3, 2, 3], 
                        [3, 3, 3, 1]])
listPreference = np.array([fuzzyfication(alternative1), fuzzyfication(alternative2), fuzzyfication(alternative3)])

listComparisonMatrix = fillAllComparisonMatrix(listComparisonMatrix)
fuzzied = aggegate(listComparisonMatrix)
show(fuzzied, "fuzzyComaprisonMatrix")

fuzzied_normalized = normalize(fuzzied)
show(fuzzied_normalized, "normalizedFuzzyComaprisonMatrix")

w = getWeightVector(fuzzied_normalized)
w = np.reshape(w, (-1, 1, 3))
w = reorder(w)
show(w, "criteriaWeightVector")

show(listPreference, "fuzzyAlternativeMatrix")
G_L = np.min(listPreference[:,:,:,0], axis=0)
G_M = np.average(listPreference[:,:,:,1], axis=0)
G_U = np.max(listPreference[:,:,:,2], axis=0)
G_matrix = np.dstack((G_L, G_M, G_U))
show(G_matrix, "G_matrix")

a_matrix = np.empty(G_matrix.shape)
for j in range(0, len(a_matrix[0])):
    sum_column_normalize = np.sqrt(np.sum(G_matrix[:,j]**2, axis=0))
    for i in range(0, len(a_matrix)):
        a_matrix[i][j] = G_matrix[i][j]/sum_column_normalize
show(a_matrix, "a_matrix")

a_matrix_reorder = reorder(a_matrix)
show(a_matrix_reorder, "a_matrix_reorder")

h_matrix = a_matrix.copy()
for i in range(0, len(h_matrix[0])):
    h_matrix[:,i] *= w[i]
show(h_matrix, "h_matrix")

h_matrix_reorder = reorder(h_matrix)
show(h_matrix_reorder, "h_matrix_reorder")

h_defuzzy = defuzzyfication(h_matrix_reorder, alpha=0.6, beta=0.5)
show(h_defuzzy, "defuzzy")

h_max = np.max(h_defuzzy, axis=0)
h_min = np.min(h_defuzzy, axis=0)
h_beta1 = h_defuzzy.copy()
h_beta2 = h_defuzzy.copy()
for i in range(0, len(h_defuzzy)):
    h_beta1[i]=h_beta1[i]-h_max
    h_beta2[i]=h_beta2[i]-h_min
show(h_beta1, "h1")
show(h_beta2, "h2")
S_max = np.sqrt(np.sum(h_beta1**2, axis=1)).T
S_min = np.sqrt(np.sum(h_beta2**2, axis=1)).T
S = np.vstack((S_max, S_min))
S = np.transpose(S, axes=(1,0))
show(S, "S_matrix")

R = S[:,1]/(S[:,1]+S[:,0])
R = normalize(R)
show(R, "R_matrix")
