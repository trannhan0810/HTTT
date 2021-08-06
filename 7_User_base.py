import numpy as np
import math
from numpy import sum, sqrt, average, array as arr, abs, arange

np.set_printoptions(formatter={'float':lambda x:"{0:0.2f}".format(x)})

X = -1
rating = arr([
    [ 1, 4, 5, X, 3 ],
    [ 5, 1, X, 5, 2 ],
    [ 4, 1, 2, 5, X ],
    [ X, 3, X, 4, 4 ]], dtype=float)

''' Input:   
      u1=[-1, 2, 3, 4, 5] 
      u2=[ 1, 7, 8, 9,-1]
    => Dtrain = [False, True, True, True, False]        
    Ountput:
      u1_filtered = [ 2, 3, 4]
      u2_filtered = [ 7, 8, 9]'''
def findIntersectSet(u1, u2):
    Dtrain = np.logical_and(u1 != X, u2 != X )
    u1_filtered = u1[Dtrain]
    u2_filtered = u2[Dtrain]
    return u1_filtered, u2_filtered
    # u1_filtered = [];
    # u2_filtered = [];
    # for i in range(len(u1)):
    #   if u1[i] != X and u2[i] != X: 
    #     u1_filtered.append(u1[i]);
    #     u2_filtered.append(u2[i]);
    # return arr(u1_filtered), arr(u2_filtered)

def simCosin(u1, u2): 
    u1, u2 = findIntersectSet(u1, u2)
    return np.dot(u1, u2)/(sqrt(sum(u1**2) * sum(u2**2)))

def simPearson(u1, u2): 
    u1_mean = average(u1[u1!=X])
    u2_mean = average(u2[u2!=X])
    # u1_mean = 0; u1_count = 0; u2_mean = 0; u2_count = 0;
    # for i in range(len(u1)):
    #   if(u1[i]!=X):
    #     u1_mean+=u1[i];
    #     u1_count+=1;
    # u1_mean = u1_mean/u1_count
    # for i in range(len(u2)):
    #   if(u2[i]!=X):
    #     u2_mean+=u2[i];
    #     u2_count+=1;
    # u2_mean = u2_mean/u2_count
    u1, u2 = findIntersectSet(u1, u2)
    return np.dot(u1-u1_mean, u2-u2_mean)/(sqrt(sum((u1-u1_mean)**2)*sum((u2-u2_mean)**2)))

def sim(u1, u2, type="cosin"):
    if type == "cosin": return simCosin(u1, u2);
    if type == "pearson": return simPearson(u1, u2);

'''
'''
def get_K_Nearest_Neighbor(user_index, item_index, k_number, rating, simArray):
    neighbor = arr([u for u in arange(rating.shape[0]) if rating[u,item_index] != X]);
    # neighbor = [];
    # for u in rating.shape[0]:
    #   if rating[u,item_index] != X]:
    #     neighbor.append(u);
    # neighbor = arr(neighbor);

    return sorted(neighbor, key = lambda x: simArray[x], reverse=True)[:k_number];

def pred(user_index, item_index, neighbor, rating, simArray, type="cosin"):
    filteredSimArray = simArray[neighbor];
    filteredRating = rating[neighbor];
    # filteredSimArray = [];
    # filteredRating = [];
    # for u in neighbor:
    #   filteredSimArray.append(simArray[u]);
    #   filteredRating.append(rating[u]);

    if type == "cosin":
        return sum(filteredSimArray*filteredRating[:,item_index])/sum(abs(filteredSimArray))
    if type == "pearson":
        u_mean = [average(u[u!=X]) for u in filteredRating]
        return u_mean[user_index] + sum(filteredSimArray*(filteredRating[:,item_index] - u_mean))/sum(abs(filteredSimArray))

def USER_BASE(rating, type = "cosin"):
    NUM_OF_USER, NUM_OF_ITEM = rating.shape
    simMatrix = np.zeros((NUM_OF_USER, NUM_OF_USER));
    for i in range(NUM_OF_USER): 
      for j in range(NUM_OF_USER): simMatrix[i][j] = sim(rating[i], rating[j], type);
    simMatrix = arr(simMatrix)
    print("similarity user-base")
    print(simMatrix)

    result = np.array(rating.copy(), dtype=float)
    for i in range(NUM_OF_USER):
      for j in range(NUM_OF_ITEM):
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

# print(rating)

# print("===============USER BASE=================")
# userbase = USER_BASE(rating, "pearson")
# print("Result user base")
# print(userbase)

# print("================ITEM BASE==================")
# itembase = ITEM_BASE(rating)
# print("Result item base")
# print(itembase)

# print("=============USER ITEM BASE===============")
# print("Result user-item base")
# x = 0.5
# print(userbase*x + itembase*(1-x))


print(sum([]))