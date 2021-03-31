import numpy as np 
np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})

#Điền đầy đủ vào các ma trận so sánh cặp
def fill(matrix):
    length = len(matrix) 
    #Set polygon = 1
    for i in range(0, length): matrix[i,i] = 1
    #If row 0 contain 0 raise an error
    if any(cell == 0 for cell in matrix[0] ): raise "missing number in row 0"
    for i in range(0, length):
        for j in range(0, length):
            if matrix[i,j] == 0:
                #If cell[j,i] have value, use inverse number of cell[j,i]
                if matrix[j,i] != 0: matrix[i,j] = 1/matrix[j,i]
                 #Else use the chain rule: cell[0,j]=cell[0,i]*cell[i,j]
                else: matrix[i,j] = matrix[0,j]/matrix[0,i]
    return matrix

#Tính vector trọng số của ma trận so sánh
def getWeightVector(matrix):
    #Copy ma trận để không ảnh hưởng ma trận ban đầu
    matrix = matrix.copy()
    #Chuẩn hóa ma trận bằng cách chia từng ô cho tổng cột tương ứng
    matrix = matrix/np.sum(matrix, axis=0)
    #Tính trọng số bằng cách tính trung bình cộng mỗi hàng
    w = np.sum(matrix, axis=1)/matrix.shape[1]
    return w

#Tổng hợp các ma trận lại thành một ma trận mờ kích thước m hàng n cột 3 giá trị mờ
def integrate(listComparisonMatrix, dim = 3):
    if dim == 3: 
        L = np.min(listComparisonMatrix, axis = 0) #Tính min của mảng
        M = np.average(listComparisonMatrix, axis = 0) #Tính trung bình của mảng
        U = np.max(listComparisonMatrix, axis = 0) #Tính max của mảng
        return np.dstack((L, M, U))
    if dim == 4:
        L = np.min(listComparisonMatrix[:,:,:,0], axis = 0)     #Tính min của min của các phần tử mảng
        M = np.average(listComparisonMatrix[:,:,:,1], axis = 0) #Tính ave của ave của các phần tử mảng
        U = np.max(listComparisonMatrix[:,:,:,2], axis = 0)     #Tính max của max của các phần tử mảng
        return np.dstack((L, M, U))

#Làm mờ hóa ma trận Vd: 1->(1,1,3); 3->(1,3,5) 
def fuzzyfication(M):
    L = np.maximum(M-2, np.full((M.shape), 1))
    U = np.minimum(M+2, np.full((M.shape), 9))
    return np.dstack((L, M, U))

#Khử mờ
def defuzzyfication(fuzzyMatrix, alpha = 0.5, beta = 0.5):
    #Xác định khoảng rõ (khử alpha)
    alpha_l = fuzzyMatrix[:,:,0] + alpha*(fuzzyMatrix[:,:,1] - fuzzyMatrix[:,:,0])
    alpha_r = fuzzyMatrix[:,:,2] - alpha*(fuzzyMatrix[:,:,2] - fuzzyMatrix[:,:,1])
    alpha_matrix = np.dstack((alpha_l, alpha_r))

    #Xác định giá trị rõ (khử beta)
    beta_matrix = alpha_l*beta + alpha_r*(1-beta)
    return  alpha_matrix, beta_matrix

#Xắp xếp lại các giá trị mờ
def reorder(matrix):
    if(matrix.ndim != 3): raise "ERROR"
    listLMU = [matrix[:,:,0], matrix[:,:,1], matrix[:,:,2]]
    L = np.min(listLMU, axis = 0)
    U = np.max(listLMU, axis = 0)
    M = np.sum(listLMU, axis = 0) - L - U
    return np.dstack((L, M, U))

def show(matrix, title="---------"):
    print("---"+title+"----")
    if matrix.ndim >4 : raise "Error, can't show matrix with high than 4 dimension"
    if matrix.ndim == 4 : [show(matrix[i]) for i in range(0,3)]; return;
    if matrix.ndim == 3 :
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                print(matrix[i][j], end=", ")
            print("")
    else: print(matrix)
#==================================================================
#Sample các ma trận so sánh các tiêu chí với nhau
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
#Sample các ma trận đánh giá phương án (hàng) theo các tiêu chí (cột)
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
D_matrix = integrate(list_criteria_comparison, dim = 3)
show(D_matrix, "listOfFuzzyComaprisonMatrix")
#Chuẩn hóa và tìm trọng số từ các ma trận so sánh cặp tiêu chí
w = getWeightVector(D_matrix).reshape((-1,1,3))
w_reorder = reorder(w)
show(w_reorder, "criteriaWeightVector")

#Làm mờ hóa các ma trận đánh giá phương án - tiêu chí
list_fuzzied_alternative_criteria = np.array([fuzzyfication(e) for e in list_alternative_criteria])
show(list_fuzzied_alternative_criteria, "fuzzyAlternativeCriteriaMatrix")
#Tích hợp các ma trận đánh giá phương án - tiêu chí
G_matrix = integrate(list_fuzzied_alternative_criteria, dim = 4)
show(G_matrix, "G_matrix")
#Chuẩn hóa ma trận đánh giá phương án - tiêu chí bằng cách chia từng phần tử cho căn bậc 2 của tổng bình phương của cột tương ứng
a_matrix = G_matrix/np.sqrt(np.sum((G_matrix**2), axis=0))
show(a_matrix, "a_matrix")
a_matrix_reorder = reorder(a_matrix)
show(a_matrix_reorder, "a_matrix_reorder")

#Tổng hợp
h_matrix = a_matrix_reorder*w_reorder.reshape((-1, 3))
show(h_matrix, "h_matrix")
# h_matrix2 = a_matrix*w.reshape((-1, 3))
# show(reorder(h_matrix2), "h_matrix")
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
R = R/np.sum(R)
show(R, "R_matrix")
