import numpy as np
from numpy.lib.shape_base import split
np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})
from scipy.optimize import linprog

n=15

num_of_cri = 2
criterias = ['C1', 'C2']
cri_comb =          ['C1',  'C2',   'C1_C2' ]
cri_mass = np.array([  6,     4,      5     ])/n

num_of_alter = 3
alternatives =  ['A1', 'A2', 'A3']
alter_comb =               [ 'A1', 'A2',  'A3', 'A1_A2', 'A1_A3', 'A2_A3', 'A1_A2_A3'  ]
alter_mass_C1 =   np.array([   5,   2,     3,      4,      0,        0,        1       ])/n
alter_mass_C2 =   np.array([   3,   1,     2,      3,      3,        1,        2       ])/n
alter_mass = np.array([ alter_mass_C1, alter_mass_C2 ])

#Tìm vị trí các tổ hợp là tập con của element trong array
#Vd element = AB, arr = [A, B, C, AB, AC, BC, ABC]  => return [ 0, 1, 3 ]
def getSubSet(element, array):
    return [i for i in range(len(array)) if all(x in element.split('_') for x in array[i].split('_'))]
#Tính Bel với mass là giá trị ứng với tổ hợp comb tương ứng
#Vd Bel[AB] = mass[A] + mass[B] + mass[AB]
def Bel(comb, mass):
    return np.array([np.sum([mass[k] for k in getSubSet(element, comb)]) for element in comb])

#Tìm vị trí các tổ hợp có giao với element trong array
#Vd element = A, arr = [A, B, C, AB, AC, BC, ABC]  => return [ 0, 3, 4, 6 ]
def getRelateSet(element, array):
    return [i for i in range(len(array)) if any(x in element.split('_') for x in array[i].split('_'))]
#Tính Pl với mass là giá trị ứng với tổ hợp comb tương ứng
#vd Pl[A] = mass[A] + mass[AB] + mass[AC] + mass[ABC]
def Pl(comb, mass):
   return np.array([np.sum([mass[k] for k in getRelateSet(element, comb)]) for element in comb])

#Covert từ mảng một tổ hợp thành một mảng gồm các số 0, 1
#vd combs = [ A, B, AB ];  names = [A, B]; => return [ [1,0], [0,1], [1,1] ]
def leftSideInequantion(combs, names):
    return np.array([[1 if(x in comb.split('_')) else 0 for x in names] for comb in combs])
#                                       left_side:          right_side:
# Bel[C1]    ≤ p1                      [[-1 , 0 ]   ≤     [-Bel[0]
# Bel[C2]    ≤ p1                       [ 0 ,-1 ]          -Bel[1]
# Bel[C1_C2] ≤ p1+p2                    [ 1 ,-1 ]          -Bel[2]
# p1         ≤ Pl[C1]              <=>  [ 1 , 0 ]           Pl[0]
# p2         ≤ Pl[C2]                   [ 0 , 1 ]           Pl[1]
# p1 + p2    ≤ Pl[C1_C2]                [ 1 , 1 ]]          Pl[2]
def getCriteriaConditional(cri_comb, criterias, criteria_Pl, criteria_Bel):
    left = leftSideInequantion(cri_comb, criterias)
    left_side = np.concatenate((-left, left))
    right_side = np.concatenate((-criteria_Bel, criteria_Pl))
    return left_side, right_side

criteria_Bel = Bel(cri_comb, cri_mass)
criteria_Pl = Pl(cri_comb, cri_mass)

alternatives_Bel = [Bel(alter_mass[key], alter_comb) for key in range(num_of_cri) ]
alternatives_Pl = [Pl(alter_mass[key], alter_comb) for key in range(num_of_cri) ]

left_side, right_side = getCriteriaConditional(cri_comb, criterias, criteria_Pl, criteria_Bel)

c_bel = [ np.array([alternatives_Bel[cri][alter] for cri in range(num_of_cri)]) for alter in range(len(alter_comb)) ]
BEL = np.array([ np.dot((linprog(c, left_side, right_side).x), c) for c in c_bel])

c_pl = [ np.array([alternatives_Pl[cri][alter] for cri in range(num_of_cri)]) for alter in range(len(alter_comb)) ]
PL = np.array([ np.dot((linprog(-c, left_side, right_side).x), c) for c in c_pl])

alpha = 0.4
final_result = alpha*BEL + (1-alpha)*PL

print("BEL:\n", BEL)
print("PL:\n", PL)
print("Final result:\n", final_result)
