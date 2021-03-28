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

#Tìm các tập con của element trong array
def getSubSet(element, array):
    return [i for i in range(len(array)) if all(x in element.split('_') for x in array[i].split('_'))]
def Bel(mass, comb):
    return np.array([np.sum([mass[k] for k in getSubSet(element, comb)]) for element in comb])

def getRelateSet(element, array):
    return [i for i in range(len(array)) if any(x in element.split('_') for x in array[i].split('_'))]
def Pl(mass, comb):
   return np.array([np.sum([mass[k] for k in getRelateSet(element, comb)]) for element in comb])

def leftSideInequantion(combs, names):
    return np.array([[1 if(x in comb.split('_')) else 0 for x in names] for comb in combs])

def getCriteriaConditional(cri_comb, criterias, criteria_Pl, criteria_Bel):
    left = leftSideInequantion(cri_comb, criterias)
    left_side = np.concatenate((left, -left))
    right_side = np.concatenate((criteria_Pl, -criteria_Bel))
    return left_side, right_side

criteria_Bel = Bel(cri_mass, cri_comb)
criteria_Pl = Pl(cri_mass, cri_comb)

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
