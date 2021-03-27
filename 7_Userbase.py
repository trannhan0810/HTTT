import numpy as np
import math

np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})

X = -1
rating =  np.array([[ 1, 4, 5, X, 3 ],
                    [ 5, 1, X, 5, 2 ],
                    [ 4, 1, 2, 5, X ],
                    [ X, 3, X, 4, 4 ]])

def sim(r1, r2, type="cosin"):
    n = len(r1)
    Dtrain = [ k for k in range(n) if r1[k]*r2[k]>=0 ]
    if type == "cosin":
        return np.sum(r1[k]*r2[k] for k in Dtrain)/(
            np.sqrt(np.sum(r1[k]**2 for k in Dtrain))*np.sqrt(np.sum(r2[k]**2 for k in Dtrain)))
    if type == "pearson":
        r1_ave = np.average([k for k in r1 if k>=0 ])
        r2_ave = np.average([k for k in r2 if k>=0 ])
        return np.sum((r1[k]-r1_ave)*(r2[k]-r2_ave) for k in Dtrain)/(
            np.sqrt(np.sum((r1[k]-r1_ave)**2 for k in Dtrain))*np.sqrt(np.sum((r2[k]-r2_ave)**2 for k in Dtrain)))


def USERKNN_CF(r, i, j, u, type="cosin"):
    simArr = [sim(r[i],r[k]) for k in range(len(r))]
    simArrFilter = [simArr[k] for k in range(r) if (r[k][j] >= 0)]
    simArrSorted = sorted(simArrFilter, reverse=True)
    U = [k for k in len(simArr) if simArr >= simArrSorted[u-1]] 
    if type == "cosin":
        return np.sum([simArr[k]*r[k][j] for k in U])/np.sum([np.abs(simArr[k]) for k in U])
    if type == "pearson":
        r_ave = [np.average([l for l in r[k] if l>=0 ]) for k in len(r)]
        return r_ave[i] + np.sum([simArr[k]*(r[k][j]-r_ave[k]) for k in U])/np.sum([np.abs(simArr[k]) for k in U])

num_of_user, num_of_item = rating.shape
simMatrix = np.zeros((num_of_user, num_of_user), dtype=float)
for i in range(num_of_user):
        for j in range(num_of_user):
            simMatrix[i][j] = sim(rating[i], rating[j])
print(simMatrix)
