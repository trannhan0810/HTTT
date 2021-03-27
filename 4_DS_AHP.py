import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{0:0.1f}".format(x)})


criteria_name = ['C1', 'C2']
criterias = { 'C1': 6, 'C2': 4, 'C1_C2': 5 }

alternatives_name = {'A1', 'A2', 'A3'}
alternatives_C1 = { 'A1': 5, 'A2': 2, 'A3': 3, 'A1_A2': 4, 'A1_A3': 0, 'A2_A3': 0, 'A1_A2_A3': 1}
alternatives_C2 = { 'A1': 3, 'A2': 1, 'A3': 2, 'A1_A2': 3, 'A1_A3': 3, 'A2_A3': 1, 'A1_A2_A3': 2}
alternatives = { 'C1': alternatives_C1, 'C2': alternatives_C2 }

def calculateBel(dictionary):
    return {key: np.sum([dictionary[k] for k in dictionary.keys() if all(x in key.split('_') for x in k.split('_'))]) for key in dictionary.keys()}

def calculatePl(dictionary):
    return {key: np.sum([dictionary[k] for k in dictionary.keys() if any(x in key.split('_') for x in k.split('_'))]) for key in dictionary.keys()}

criteria_Bel = calculateBel(criterias)
criteria_Pl = calculatePl(criterias)

alternatives_Bel = { key: calculateBel(alternatives[key]) for key in criteria_name }
alternatives_Pl = { key: calculatePl(alternatives[key]) for key in criteria_name }
