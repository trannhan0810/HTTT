import numpy as np
from numpy import array as arr
from numpy.lib.shape_base import split
np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})
from scipy.optimize import linprog

n=15

NUM_OF_CRITERIA = 2; 
NUM_OF_CRITERIA_COMBINATION = 3; 
criterias = ['C1', 'C2']
cri_comb = np.array([['C1'],  ['C2'],   ['C1','C2'] ], dtype= object)
cri_mass = np.array([  6,     4,      5     ])/n

NUM_OF_ALTERNATIVE = 3
NUM_OF_ALTERNATIVE_COMBINATION = 7
alternatives =  ['A1', 'A2', 'A3']
alter_comb =    np.array([ ['A1'], ['A2'],  ['A3'], ['A1','A2'], ['A1','A3'], ['A2','A3'], ['A1','A2','A3']  ], dtype=object)
alter_mass_C1 = np.array([   5,   2,     3,      4,      0,        0,        1       ])/n
alter_mass_C2 = np.array([   3,   1,     2,      3,      3,        1,        2       ])/n
alter_mass = np.array([ alter_mass_C1, alter_mass_C2 ])

# Vd comb1 = ['A1','A2']
#    comb2 = ['A2']
#    comb3 = ['A2','A3']
# isSub(comb2, comb1) = True, isSUb(comb3, comb1) = False 
def isSub(comb1, comb2):  
    for element in comb1: 
        if(not (element in comb2)): 
            return False
    return True

#Tìm vị trí các tổ hợp là tập con của element trong array
#Vd: comb =  ['A1','A2'],
#    array_of_comb = [ ['A1'], ['A2'],  ['A3'], ['A1','A2'], ['A1','A3'], ['A2','A3'], ['A1','A2','A3']  ]
# => return index of [[A1], ['A2'], ['A1','A2']] = [0, 1, 3]
def getSubSet(comb, array_of_comb):
    return [i for i in np.arange(len(array_of_comb)) if isSub(array_of_comb[i], comb)]
    
#Tính Bel với mass là giá trị ứng với tổ hợp comb tương ứng
#Vd Bel[AB] = mass[A] + mass[B] + mass[AB]
def Bel(comb_arr, mass_arr):
    Bel_list = np.zeros(len(comb_arr))
    for i in range(len(comb_arr)):
        # comb_arr = [ ['A1'], ['A2'],  ['A3'], ['A1','A2'], ['A1','A3'], ['A2','A3'], ['A1','A2','A3']  ]
        # mass_arr = [    5,     2,       3,         4,           0,            0,            1          ]
        # i = 3
        # comb_arr[i] = ['A1','A2']
        # getSubSet(comb_arr[i], comb_arr) = [0, 1, 3]
        # mass_arr[getSubSet(comb_arr[i], comb_arr)] = [5, 2, 4]
        # Bel_list[3] = 11
        Bel_list[i] = np.sum(mass_arr[getSubSet(comb_arr[i], comb_arr)])
    return arr(Bel_list)

    ##return np.array([np.sum([mass[getSubSet(element, comb)]]) for element in comb])

# Vd comb1 = ['A1','A2']
#    comb2 = ['A3']
#    comb3 = ['A2','A3']
# isSub(comb2, comb1) = False, isSUb(comb3, comb1) = True
def isRelate(comb1, comb2):
    for element in comb1: 
        if((element in comb2)): 
            return True
    return False

#Tìm vị trí các tổ hợp có giao với element trong array
#Vd: comb =  ['A1','A2'],
#    array_of_comb = [ ['A1'], ['A2'],  ['A3'], ['A1','A2'], ['A1','A3'], ['A2','A3'], ['A1','A2','A3']  ]
# => return index of [[A1], ['A2'], ['A1','A2'], ['A1','A3'], ['A2','A3'], ['A1','A2','A3'] ] = [0, 1, 3, 4, 5, 6]
def getRelateSet(comb, array_of_comb):
       return [i for i in np.arange(len(array_of_comb)) if isRelate(array_of_comb[i], comb)]

#Tính Pl với mass là giá trị ứng với tổ hợp comb tương ứng
#vd Pl[A] = mass[A] + mass[AB] + mass[AC] + mass[ABC]
def Pl(comb_arr, mass_arr):
    Pl_list = np.zeros(len(comb_arr))
    for i in range(len(comb_arr)):
        # comb_arr = [ ['A1'], ['A2'],  ['A3'], ['A1','A2'], ['A1','A3'], ['A2','A3'], ['A1','A2','A3']  ]
        # mass_arr = [    5,     2,       3,         4,           0,            0,            1          ]
        # i = 3
        # comb_arr[i] = ['A1','A2']
        # getRelateSet(comb_arr[i], comb_arr) = [0, 1, 3, 4, 5, 6]
        # mass_arr[getSubSet(comb_arr[i], comb_arr)] = [5, 2, 4, 0, 0, 1]
        # Bel_list[3] = 12
        Pl_list[i] = np.sum(mass_arr[getRelateSet(comb_arr[i], comb_arr)])
    return arr(Pl_list)

#Covert từ mảng một tổ hợp thành một mảng gồm các số 0, 1
#vd comb_arr = [ [A1], [A2], [A1, A2] ];  names = [A1,A2]; => return [ [1,0], [0,1], [1,1] ]
def leftSideInequantion(comb_arr, names):
    result = []
    for comb in comb_arr:
        encode = []
        for name in names:
            if name in comb: encode.append(1)
            else: encode.append(0)
        result.append(encode)
    return arr(result) 

#                                       left_side:          right_side:
# -p1        ≤ -Bel[C1]                [[-1 , 0 ]   ≤     [-Bel[0]
# -p2        ≤ -Bel[C2]                 [ 0 ,-1 ]          -Bel[1]
# -p1-p2     ≤ -Bel[C1_C2]              [-1 ,-1 ]          -Bel[2]
# p1         ≤ Pl[C1]              <=>  [ 1 , 0 ]           Pl[0]
# p2         ≤ Pl[C2]                   [ 0 , 1 ]           Pl[1]
# p1 + p2    ≤ Pl[C1_C2]                [ 1 , 1 ]]          Pl[2]]
def getCriteriaConditional(cri_comb, criterias, criteria_Pl, criteria_Bel):
    # left =[[ 1 , 0 ]
    #        [ 0 , 1 ]
    #        [ 1 , 1 ]]
    left = leftSideInequantion(cri_comb, criterias)

    # left_side =[[-1 , 0 ]
    #             [ 0 ,-1 ]
    #             [-1 ,-1 ]          
    #             [ 1 , 0 ]          
    #             [ 0 , 1 ]           
    #             [ 1 , 1 ]]          
    left_side = np.concatenate((-left, left))

    #right_sile = [-Bel[0], -Bel[1], -Bel[2], Pl[0], Pl[1], Pl[2]]
    right_side = np.concatenate((-criteria_Bel, criteria_Pl))
    return left_side, right_side

# Calculate Bel, Pl for criteria
criteria_Bel = Bel(cri_comb, cri_mass)  
criteria_Pl = Pl(cri_comb, cri_mass)

# Calculate Bel, Pl for alternative
alternatives_Bel = []; alternatives_Pl = []
for criteria in range(NUM_OF_CRITERIA):
    alternatives_Bel.append( Bel(alter_comb, alter_mass[criteria]) )
    alternatives_Pl.append( Pl(alter_comb, alter_mass[criteria]) )
alternatives_Bel = arr(alternatives_Bel); 
alternatives_Pl = arr(alternatives_Pl)

print("criteria_Bel: \n", criteria_Bel)  
#=> [0.400 0.267 1.000]
print("criteria_Pl: \n", criteria_Pl)    
#=> [0.733 0.600 1.000]
print("alternatives_Bel: \n", alternatives_Bel)
#=>[[0.333 0.133 0.200 0.733 0.533 0.333 1.000]
#   [0.200 0.067 0.133 0.467 0.533 0.267 1.000]]
print("alternatives_pl: \n", alternatives_Pl)
#=> [[0.667 0.467 0.267 0.800 0.867 0.667 1.000]
#    [0.733 0.467 0.533 0.867 0.933 0.800 1.000]]
left_side, right_side = getCriteriaConditional(cri_comb, criterias, criteria_Pl, criteria_Bel)

BEL = np.zeros(NUM_OF_ALTERNATIVE_COMBINATION); 
PL = np.zeros(NUM_OF_ALTERNATIVE_COMBINATION);
for i in range(NUM_OF_ALTERNATIVE_COMBINATION):
    #return [p1, p2, ...] where p1*alternatives_Bel[0,i] + p2*alternatives_Bel[1,i] + ... ---> Min
    optimize_C_bel = linprog(alternatives_Bel[:,i], left_side, right_side).x
    BEL[i] = np.dot(optimize_C_bel, alternatives_Bel[:,i])
     #return [p1, p2, ...] where p1*(-alternatives_Bel[0,i]) + p2*(-alternatives_Bel[1,i]) + ... ---> Min
    optimize_C_pl = linprog(-alternatives_Pl[:,i], left_side, right_side).x 
    PL[i] = np.dot(optimize_C_pl,  alternatives_Pl[:,i])

alpha = 0.4
final_result = alpha*BEL + (1-alpha)*PL

print("BEL:\n", BEL)                    #=> [0.253 0.093 0.160 0.573 0.533 0.293 1.000]
print("PL:\n", PL)                      #=> [0.707 0.467 0.427 0.840 0.907 0.747 1.000]
print("Final result:\n", final_result)  #=> [0.525 0.317 0.320 0.733 0.757 0.565 1.000]
