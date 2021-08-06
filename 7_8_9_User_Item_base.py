import numpy as np
import math
from numpy import sum, sqrt, average, array as arr, abs

np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})

X = -1
rating = np.array([
    [ 1, 4, 5, X, 3 ],
    [ 5, 1, X, 5, 2 ],
    [ 4, 1, 2, 5, X ],
    [ X, 3, X, 4, 4 ]], dtype=float)

'''Input: 
u1:[ 1, 2, 3, 4, 5] => u1_filtered = [ 2, 3, 4]
u2:[-6, 7, 8, 9, -1] => u2_filtered = [ 7, 8, 9]
Dtrain = [False, True, True, True, False]'''
def findIntersectSet(u1, u2):
    #u1: [ 1, 2, 3, 4, 5] => u1>=0 : [ True, True, True, True, True]
    #u1: [-6, 7, 8, 9, -1] => u2>=0 : [ False, True, True, True, False]
    Dtrain = np.logical_and(u1>=0, u2 >=0 )
    u1_filtered = u1[Dtrain]
    u2_filtered = u2[Dtrain]
    return u1_filtered, u2_filtered

def simCosin(u1, u2): 
    u1, u2 = findIntersectSet(u1, u2)
    return np.dot(u1, u2)/(
        sqrt(sum(u1**2) * sum(u2**2)))

def simPearson(u1, u2): 
    u1_mean = average(u1[u1>=0])
    u2_mean = average(u2[u2>=0])
    u1, u2 = findIntersectSet(u1, u2)
    print((u1,u2))
    return np.dot(u1-u1_mean, u2-u2_mean)/(
        sqrt(sum((u1-u1_mean)**2)*sum((u2-u2_mean)**2)))

def sim(u1, u2, type="cosin"):
    if type == "cosin": 
        return simCosin(u1, u2)
    else: 
        return simPearson(u1, u2)

def get_K_Nearest_Neighbor(user_index, item_index, k, rating, simArray):
    #rating.shape = (4, 5) => 4 user, 5 item
    NUM_OF_USER = len(rating)
    neighbor = np.arange(NUM_OF_USER) #=> neighbor = [0,1,2,3]
    #Loai nhung thang ko danh gia item item_index
    neighbor = neighbor[ (lambda x: rating[x, item_index] >= 0)(neighbor)]
    #Loai ra thang user co index = user_index
    neighbor = neighbor[ neighbor != user_index ]
    neighbor = sorted(neighbor, key = lambda x: simArray[x], reverse=True)[:k]
    return neighbor 



def pred(user_index, item_index, neighbor, rating, simArray, type="cosin"):
    #Vd simArray = [ 0.9, 1, 0.8, 0.5], neighbor = [0, 2] => simArray[neighbor] = [0.9,0.8]
    simArray = simArray[ neighbor ]
    #vd rating=[[ 1, 4, 5, X, 3 ],    neighbor = [0, 2] => rating[neighbor] =[[ 1, 4, 5, X, 3 ]
    #           [ ?, 1, X, 5, 2 ],                                            [ 4, 1, 2, 5, X ]]
    #           [ 4, 1, 2, 5, X ],
    #           [ X, 3, X, 4, 4 ]]
    rating = rating[ neighbor ]

    if type == "cosin":
        #Vd item_index = 0 
        #vd user_index = 1
        #=> rating[1,0] = [0.9*1 + 0.8*4]/(0.9+0.8) = ...
        #=> rating[1,3] = []
        return np.sum(simArray*rating[:,item_index])/sum(abs(simArray))
    if type == "pearson":
        u_mean = [ average(u[u>=0]) for u in rating]
        return u_mean[user_index] + np.sum(simArray*(rating[:,item_index] - u_mean))/np.sum(np.abs(simArray))

def USER_BASE(rating, type = "cosin"):
    num_of_user, num_of_item = rating.shape

    simMatrix = np.array([ sim(rating[i], rating[j], type) for i in range(num_of_user) for j in range(num_of_user)])\
        .reshape((num_of_user,num_of_user))
    print("similarity user-base")
    print(simMatrix)

    result = np.array(rating.copy(), dtype=float)
    for i in range(num_of_user):
        for j in range(num_of_item):
            if result[i][j] == X:
                neighbor = get_K_Nearest_Neighbor(i, j, k = 2, rating = rating, simArray = simMatrix[i])
                result[i][j] = pred(i, j, neighbor, rating, simMatrix[i], type)
    return result

def ITEM_BASE(rating, type = "cosin"):
    num_of_user, num_of_item = rating.shape
    rating = np.transpose(rating)
    simMatrix = np.array([ sim(rating[i], rating[j], type) for i in range(num_of_item) for j in range(num_of_item)])\
        .reshape((num_of_item, num_of_item))
    print("similarity item-base")
    print(simMatrix)

    result = np.array(rating.copy(), dtype=float)
    for i in range(num_of_item):
        for j in range(num_of_user):
            if result[i][j] == X:
                neighbor = get_K_Nearest_Neighbor(i, j, k = 2, rating = rating, simArray = simMatrix[i])
                result[i][j] = pred(i, j, neighbor, rating, simMatrix[i], type)
    return np.transpose(result)

def USER_ITEM_BASE(rating, type = "cosin"):
    alpha = 0.5
    return USER_BASE(rating, type)*alpha + ITEM_BASE(rating, type)*(1-alpha)

print(rating)

print("===============USER BASE=================")
userbase = USER_BASE(rating, "pearson")
print("Result user base")
print(userbase)

# print("================ITEM BASE==================")
# itembase = ITEM_BASE(rating)
# print("Result item base")
# print(itembase)

# print("=============USER ITEM BASE===============")
# print("Result user-item base")
# x = 0.5
# print(userbase*x + itembase*(1-x))


