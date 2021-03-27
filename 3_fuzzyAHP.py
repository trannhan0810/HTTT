import numpy as np 

np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})


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
    matrix = comparisonMatrix.copy()
    return matrix/np.sum(matrix, axis=0)

def getWeightVector(matrix):
    w = np.sum(matrix, axis=1)
    return w/matrix.shape[1]

def aggregate(listComparisonMatrix, dim = 3):
    if dim == 3:
        L = np.min(listComparisonMatrix, axis = 0)
        M = np.average(listComparisonMatrix, axis = 0)
        U = np.max(listComparisonMatrix, axis = 0)
        return np.dstack((L, M, U))
    if dim == 4:
        L = np.min(listComparisonMatrix[:,:,:,0], axis = 0)
        M = np.average(listComparisonMatrix[:,:,:,1], axis = 0)
        U = np.max(listComparisonMatrix[:,:,:,2], axis = 0)
        return np.dstack((L, M, U))

def fuzzyfication(M):
    L = np.maximum(M-2, np.full((M.shape), 1))
    U = np.minimum(M+2, np.full((M.shape), 9))
    return np.dstack((L, M, U))

def defuzzyfication(fuzzyMatrix, alpha = 0.5, beta = 0.5):
    alpha_l = fuzzyMatrix[:,:,0] + alpha*(fuzzyMatrix[:,:,1] - fuzzyMatrix[:,:,0])
    alpha_r = fuzzyMatrix[:,:,2] - alpha*(fuzzyMatrix[:,:,2] - fuzzyMatrix[:,:,1])

    alpha_matrix = np.dstack((alpha_l, alpha_r))
    beta_matrix = alpha_l*beta + alpha_r*(1-beta)
    return  alpha_matrix, beta_matrix

def reorder(matrix):
    if(matrix.ndim != 3): print("ERROR"); return;
    L = np.min([matrix[:,:,0], matrix[:,:,1], matrix[:,:,2]], axis = 0)
    U = np.max([matrix[:,:,0], matrix[:,:,1], matrix[:,:,2]], axis = 0)
    M = matrix[:,:,0] + matrix[:,:,1] + matrix[:,:,2] - L - U
    return np.dstack((L, M, U))

def show(matrix, name="---------"):
    print("---"+name+"----")
    if matrix.ndim >4 : raise "Error, can't show matrix with high than 4 dimension"
    if matrix.ndim == 4 : [show(matrix[i]) for i in range(0,3)]; return;
    if matrix.ndim == 3 :
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                print(matrix[i][j], end=", ")
            print("")
    else: print(matrix)
#==================================================================
#Định nghĩa các ma trận đánh giá phương án (hàng) theo các tiêu chí (cột)
list_criteria_comparison = np.array([
        #Chuyên gia 1
        [[0.0, 4.0, 3.0, 2.00], 
         [0.0, 0.0, 0.8, 0.70], 
         [0.0, 0.0, 0.0, 0.60], 
         [0.0, 0.0, 0.0, 0.00]], 
        #Chuyên gia 2
        [[0.0, 4.2, 3.0, 1.50], 
         [0.0, 0.0, 0.85,0.70], 
         [0.0, 0.0, 0.0, 0.65], 
         [0.0, 0.0, 0.0, 0.00]], 
        #Chuyên gia 3
        [[0.0, 3.9, 2.8, 2.30], 
         [0.0, 0.0, 0.7, 0.70], 
         [0.0, 0.0, 0.0, 0.65], 
         [0.0, 0.0, 0.0, 0.00]]
], dtype=float)
#Định nghĩa các ma trận đánh giá phương án (hàng) theo các tiêu chí (cột)
list_alternative_criteria = np.array([
        #Chuyên gia 1
        [[3, 1, 5, 7], 
         [7, 5, 1, 3], 
         [5, 3, 3, 3]],
        #Chuyên gia 2
        [[5, 5, 1, 5], 
         [7, 1, 3, 5], 
         [7, 7, 1, 1]],
        #Chuyên gia 3
        [[3, 3, 1, 5], 
         [3, 5, 3, 5], 
         [5, 5, 5, 1]],
], dtype=float)

#Điền kín các ma trận so sánh cặp tiêu chí
list_criteria_comparison = [fill(e) for e in list_criteria_comparison]
#Tích hợp các ma trận so sánh cặp tiêu chí
D_matrix = aggregate(list_criteria_comparison, dim = 3)
show(D_matrix, "fuzzyComaprisonMatrix")
#Chuẩn hóa và tìm trọng số từ các ma trận so sánh cặp tiêu chí
normalied_D_matrix = normalize(D_matrix)
show(normalied_D_matrix, "normalizedFuzzyComaprisonMatrix")
w = getWeightVector(normalied_D_matrix).reshape((-1,1,3))
w = reorder(w)
show(w, "criteriaWeightVector")

#Làm mờ hóa các ma trận đánh giá phương án - tiêu chí
list_fuzzied_alternative_criteria = np.array([fuzzyfication(e) for e in list_alternative_criteria])
show(list_fuzzied_alternative_criteria, "fuzzyAlternativeCriteriaMatrix")
#Tích hợp các ma trận đánh giá phương án - tiêu chí
G_matrix = aggregate(list_fuzzied_alternative_criteria, dim = 4)
show(G_matrix, "G_matrix")
#Chuẩn hóa ma trận đánh giá phương án - tiêu chí
a_matrix = G_matrix/np.sqrt(np.sum((G_matrix**2), axis=0))
show(a_matrix, "a_matrix")
a_matrix_reorder = reorder(a_matrix)
show(a_matrix_reorder, "a_matrix_reorder")

#Tổng hợp
h_matrix = a_matrix_reorder.copy()
for i in range(0, len(h_matrix[0])):
    h_matrix[:,i] *= w[i]
#h_matrix = a_matrix*w.reshape((-1, 3))
show(h_matrix, "h_matrix")
#Khử mờ
_, h_defuzzy = defuzzyfication(h_matrix, alpha=0.6, beta=0.5)
show(h_defuzzy, "defuzzy")
#Xây dựng ma trận trung gian
h_max = np.max(h_defuzzy, axis=0) #Tính max mỗi cột
h_min = np.min(h_defuzzy, axis=0) #Tính min mỗi cột
S_max = np.sqrt(np.sum((h_defuzzy- h_max)**2, axis=1)).reshape(-1, 1)
S_min = np.sqrt(np.sum((h_defuzzy- h_min)**2, axis=1)).reshape(-1, 1)
S = np.hstack((S_max, S_min))
show(S, "S_matrix")
#Tính kết quả cuối cùng
R = S[:,1]/(S[:,1]+S[:,0])
R = normalize(R)
show(R, "R_matrix")


a = np.array([[[1, 2, 3], [4, 5, 6]],
     [[7, 8, 9], [10,11,12]]])
w =  np.array([[[1, 10, 100]], [[2, 20, 200]]])

h = a.copy()
for i in range(0, len(h[0])):
    h[:,i] *= w[i]
show(h)
show(a*w.reshape((-1, 3)))