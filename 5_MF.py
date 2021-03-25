import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{0:0.1f}".format(x)})


# sinh ra ma tran goc
x_length = 5
y_length = 5

x = np.random.rand(x_length)
y = np.random.rand(y_length)
m = np.random.rand()

A = np.zeros((x_length,y_length))
for i in range(x_length):
    for j in range(y_length):
        A[i][j] = (0 if np.random.rand() < 0.5 else 1)*(x[i]*3+2*y[j])
print("==============================================")
print(A)

#ap dung phẩn ma tran
w=np.ones((x_length,3))                               # w.shape = (x_length, k_max)
h=np.ones((3,y_length))                               # h.shape = (k_max, y_length)

print("=================================================")
for epoch in range(3000):                      #epoch: so buoc lap
    for i in range(x_length):                         #i = 1 -> x_length
        for j in range(y_length):                     #j = 1 -> y_length
            if A[i][j] != 0:                   #k = 1 -> k_max = 3
                E = w[i,:].dot(h[:,j])-A[i][j] # gradient decent a.k.a dao_ham_ham_lost
                w[i,:] -= 0.01*E*h[:,j]  # w(i) -= beta*dao_ham_ham_lost*h(j)
                h[:,j] -= 0.01*E*w[i,:]  # h(j) -= beta*dao_ham_ham_lost*w(i)

Ans = w.dot(h)
print(Ans)