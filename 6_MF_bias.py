import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{0:0.1f}".format(x)})


# sinh ra ma tran goc
x_length = 5
y_length = 5

x_weight = np.random.rand(x_length)
y_weight = np.random.rand(y_length)
bias_row_init = np.random.rand(x_length)
bias_col_init = np.random.rand(y_length)
bias_all = np.random.rand()

A_full = np.zeros((x_length,y_length))
for i in range(x_length):
    for j in range(y_length):
        A_full[i][j] = x_weight[i]*3+2*y_weight[j] + bias_row_init[i] + bias_col_init[j] + bias_all

print("==============================================")
A = np.zeros((x_length,y_length))
for i in range(x_length):
    for j in range(y_length):
        A[i][j] = (0 if np.random.rand() < 0.3 else 1)*A_full[i][j]

print("=================================================")
#Bây h mới vào MF bias 

#Tính các giá trị ban đầu cho các bias
u = np.sum(A)/len(np.flatnonzero(A))
bias_row = [ np.sum(A[i]-u)/len(np.flatnonzero(A[i])) for i in range(x_length) ]
bias_col = [ np.sum(A[:,j]-u)/len(np.flatnonzero(A[:,j])) for j in range(y_length) ]

#Khởi tạo 2 ma trận W và H
w=np.ones((x_length,3))                              
h=np.ones((3,y_length))                             

beta = 0.01
lamb = 0.005

#Bắt đầu train
for epoch in range(40000//len(np.flatnonzero(A))):      #Update 40000 lần      
    #Duyệt qua các phần tử có A[i][j] > 0            
    for i in range(x_length):                         
        for j in range(y_length):                     
            if A[i][j] != 0:                            
                #Tính eij, lost = 1/2*eij^2
                E = A[i][j] - ( w[i,:].dot(h[:,j]) + u + bias_row[i] + bias_col[j] ) 
                #Update các bias
                u = u + beta*E
                bias_row[i] = bias_row[i] + beta*(E - lamb*bias_row[i])
                bias_col[j] = bias_col[j] + beta*(E - lamb*bias_col[j])
                #Update hàng i của W và cột j của H
                temp_w = w[i,:] + beta*(2*E*h[:,j]  -lamb*w[i,:])
                temp_h = h[:,j] + beta*(2*E*w[i,:]  -lamb*h[:,j])
                w[i,:] = temp_w
                h[:,j] = temp_h

#Ans = w.dot(h) + u 
# print("+++++++++++++++++++++++++++++++++")
# print("u");print(u)
# print("w");print(w)
# print("h");print(h)
# print("bi");print(bias_row)
# print("bj");print(bias_col)

#Tính toán kết quả cuối cùng
Ans = np.zeros((x_length, y_length))
for i in range(x_length):             
    for j in range(y_length):
        Ans[i][j] = w[i,:].dot(h[:,j]) + u + bias_row[i] + bias_col[j]

#Ma trận ban đầu
print(A_full)  
#Ma trận bị thiếu
print(A)
#Số lượng các ô bị thiếu
print("Missing: ", len(A)*len(A[0]) - len(np.flatnonzero(A)))
#Kết quả cuối cùng
print(Ans)
#Hàm loss
print("Lost1: ", np.sum((Ans - A_full)**2))

lost2 =0 
for 