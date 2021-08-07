import numpy as np
import math
from numpy import negative, sum, sqrt, average, array as arr, abs, arange

np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})

X = -9999
rating = arr([[ 1, 4, 5, X, 3 ],  
              [ 5, 1, X, 4, 2 ],              
              [ 4, 1, 2, 4, X ],             
              [ X, 3, X, 5, 4 ]], dtype=float)

# Input:      

def findIntersectSet(row1, row2):
    # row1 = [ 1, 4, 5, X, 3] 
    # row2 = [ 5, 1, X, 4, 2]  
    Dtrain = np.logical_and(row1 != X, row2 != X )
    # Dtrain = [True, True, False, False, True]
    return row1[Dtrain], row2[Dtrain]
    # row1[Dtrain] = [ 1, 4, 3]
    # row2[Dtrain] = [ 5, 1, 2]

def getMeanArr(rating): 
    return arr([average(u[u!=X]) for u in rating])
    #rating =  [[ 1, 4, 5, X, 3 ],  => return  [(1+4+5+3)/4, = [3.25,
    #           [ 5, 1, X, 4, 2 ],              (5+1+4+2)/4,    3.00,
    #           [ 4, 1, 2, 4, X ],              (4+1+2+4)/4,    2.75
    #           [ X, 3, X, 5, 4 ]]              (3+4+4)/3]      4.00]

def simCosin(u1, u2, rating): 
    # u1 = 0, u2 = 1
    row1, row2 = findIntersectSet(rating[u1], rating[u2])
    # row1 = [ 1, 4, 3];   row2 = [ 5, 1, 2]
    return np.dot(row1, row2)/(sqrt(sum(row1**2) * sum(row2**2)))
    #return (1*5+4*1+3*2)/sqrt((1+4+3)**2 + (5+1+2)**2)

def simPearson(u1, u2, rating, mean): 
    # u1 = 0, u2 = 1
    mean1 = mean[u1];  mean2 = mean[u2];
    row1, row2 = findIntersectSet(rating[u1], rating[u2])
    # row1 = [ 1, 4, 3]; avg1 = 3.25;
    # row2 = [ 5, 1, 2]; avg2 = 3.00
    row1 = row1 - mean1; row2 = row2 - mean2;
    # row1 = [ -2.25,  0.75, -0.25]; 
    # row2 = [  2.00, -2.00, -1.00]; 
    return np.dot(row1, row2)/(sqrt(sum(row1**2) * sum(row2**2)))
    #return (-2.25*2.00+0.75*(-2.00)+(-0.25)*(-1.00))/sqrt((-2.25 + 0.75 -0.25)**2 + ( 2.00 -2.00 -1.00)**2)

def get_K_Nearest_Neighbor(user, item, k_number, rating, sim):
    neighbor = arr([u for u in arange(rating.shape[0]) if rating[u, item] != X]);
    neighbor = arr([u for u in neighbor if u != user])
    #  rating=   [[ 1, 4, 5, X, 3 ],
    #             [ 5, 1, X, 4, 2 ],
    #             [ 4, 1, 2, 4, X ],
    #             [ X, 3, X, 5, 4 ]]
    # u=0, i=3 => neighbor = [1, 2, 3]
    # u=1, i=3 => neighbor = [0, 2]
    return sorted(neighbor, key = lambda i: sim[user][i], reverse=True)[:k_number];

def predCosin(user, item, rating, sim, k_number = 2):
    neighbor = get_K_Nearest_Neighbor(user, item, k_number, rating, sim)
    sim_neighbor = sim[user][neighbor];
    rating_neighbor = rating[neighbor];
    return sum(sim_neighbor*rating_neighbor[:,item])/sum(abs(sim_neighbor))

def predPearson(user, item, rating, sim, mean, k_number = 2):
    neighbor = get_K_Nearest_Neighbor(user, item, k_number, rating, sim)
    sim_u_neighbor = sim[user][neighbor];
    rating_neighbor = rating[neighbor];
    mean_neighbor = mean[neighbor]
    return mean[user] + sum(np.dot(sim_u_neighbor,(rating_neighbor[:,item] - mean_neighbor)))/sum(abs(sim_u_neighbor))

def sim(u1, u2, rating, mean, type="cosin"):
    if type == "cosin": return simCosin(u1, u2, rating);
    if type == "pearson": return simPearson(u1, u2, rating, mean);

def pred(user, item, rating, sim, mean, type="cosin"):
    if type == "cosin": return predCosin(user, item, rating, sim, k_number = 2);
    if type == "pearson": return predPearson(user, item, rating, sim, mean, k_number = 2);

def USER_BASE(rating, type = "cosin"):
    NUM_OF_USER, NUM_OF_ITEM = rating.shape
    mean = getMeanArr(rating)

    normalize_rating = rating.copy()
    for u in range(NUM_OF_USER):
        for i in range(NUM_OF_ITEM):
            if rating[u][i] != X:
                normalize_rating[u][i] = rating[u][i] - mean[u]
            else:
                normalize_rating[u][i] = 0
    print(normalize_rating)
    simMatrix = np.zeros((NUM_OF_USER, NUM_OF_USER));
    for u1 in range(NUM_OF_USER): 
        for u2 in range(NUM_OF_USER): 
            simMatrix[u1][u2] = sim(u1, u2, rating, mean, type);
    print("similarity user-base: \n", simMatrix)

    result = rating.copy();
    for u in range(NUM_OF_USER):
      for i in range(NUM_OF_ITEM):
        if result[u][i] == X:
          result[u][i] = pred(u, i, rating, simMatrix, mean, type)
    return result

def ITEM_BASE(rating, type = "cosin"):
    rating = np.transpose(rating)
    NUM_OF_ITEM, NUM_OF_USER = rating.shape
    mean = getMeanArr(rating)
    normalize_rating = arr([rating[u] - mean[u] for u in range(NUM_OF_ITEM) ])
    print(normalize_rating)
    simMatrix = np.zeros((NUM_OF_ITEM, NUM_OF_ITEM));
    for i1 in range(NUM_OF_ITEM): 
        for i2 in range(NUM_OF_ITEM): 
            simMatrix[i1][i2] = sim(i1, i2, rating, mean, type);
    print("similarity item-base: \n", simMatrix)

    result = rating.copy();
    for i in range(NUM_OF_ITEM):
        for u in range(NUM_OF_USER):
           if result[i][u] == X:
              result[i][u] = pred(i, u, rating, simMatrix, mean, type)
    return np.transpose(result)

def USER_ITEM_BASE(rating, type = "cosin"):
    alpha = 0.5
    return USER_BASE(rating, type)*alpha + ITEM_BASE(rating, type)*(1-alpha)

# print(rating)

print("===============USER BASE=================")
userbase = USER_BASE(rating, "pearson")
print("Result user base")
print(userbase)

# print("================ITEM BASE==================")
# itembase = ITEM_BASE(rating, "pearson")
# print("Result item base")
# print(itembase)

# print("=============USER ITEM BASE===============")
# print("Result user-item base")
# x = 0.5
# print(userbase*x + itembase*(1-x))

