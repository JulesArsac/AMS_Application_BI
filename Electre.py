import pandas as pd
import numpy as np

def generate_preference_table(attributes, min_max, weights):
    """
    Generates the preference table for the problem.

    Args:
        attributes: The attributes on which the data is being valued
        min_max: Array of either 1 for max or -1 for min
        weights: Weight of each attribute

    Returns:
        pref_table : The preference table of the given values.
    """

    pref_table = np.zeros((len(attributes),len(attributes)))
    for i in range(len(attributes)):
        for j in range(len(attributes[i])):
            for k in range(len(attributes)):
                if k == i:
                    pref_table[i][k] = 0
                    continue
                if (attributes[i][j]*min_max[j] > attributes[k][j]*min_max[j]):
                    pref_table[i][k] += weights[j]
    return pref_table

classes = ["A","B","C"]
attributes = [[15,16,16],
              [16,8,7],
              [13,18,12]]
min_max = [1,1,1]
weights = [0.6,0.3,0.1]

pref_table = generate_preference_table(attributes,min_max,weights)

print(pref_table)