import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{0:0.1f}".format(x)})


# sinh ra ma tran goc
x_length = 5
y_length = 5

x_weight = np.random.rand(x_length)
y_weight = np.random.rand(y_length)
bias = np.random.rand()

A = np.zeros((x_length,y_length))
for i in range(x_length):
    for j in range(y_length):
        A[i][j] = (0 if np.random.rand() < 0.5 else 1)*(x_weight[i]*3+2*y_weight[j] + bias)
print("==============================================")
print(A)

print("=================================================")
u = np.sum(A)/len(np.flatnonzero(A))
bias_row = [ np.sum(A[i]-u)/len(np.flatnonzero(A[i])) for i in range(x_length) ]
bias_col = [ np.sum(A[:,j]-u)/len(np.flatnonzero(A[:,j])) for j in range(y_length) ]

w=np.ones((x_length,3))                               # w.shape = (x_length, k_max)
h=np.ones((3,y_length))                               # h.shape = (k_max, y_length)

beta = 0.01
lamb = 0.005

for epoch in range(3000//len(np.flatnonzero(A))):                      
    for i in range(x_length):                         
        for j in range(y_length):                     
            if A[i][j] != 0:                
                E = A[i][j] - ( w[i,:].dot(h[:,j]) + u + bias_row[i] + bias_col[j] )
                u = u + beta*E
                bias_row[i] = bias_row[i] + beta*(E - lamb*bias_row[i])
                bias_col[j] = bias_col[j] + beta*(E - lamb*bias_col[j])
                w[i,:] += beta*(2*E*h[:,j]  -lamb*w[i,:])
                h[:,j] += beta*(2*E*w[i,:]  -lamb*h[:,j])

#Ans = w.dot(h) + u 
# print("+++++++++++++++++++++++++++++++++")
# print("u");print(u)
# print("w");print(w)
# print("h");print(h)
# print("bi");print(bias_row)
# print("bj");print(bias_col)

Ans = np.zeros((x_length, y_length))
for i in range(x_length):                         
    for j in range(y_length):
        Ans[i][j] = w[i,:].dot(h[:,j]) + u + bias_row[i] + bias_col[j]
print(Ans)