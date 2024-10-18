import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def concordance(attributes, min_max, weights):
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
                if (attributes[i][j]*min_max[j] >= attributes[k][j]*min_max[j]):
                    pref_table[i][k] += weights[j]
    return pref_table

def no_discordance(attributes, min_max, veto):
    no_discordance_table = np.ones((len(attributes),len(attributes)))
    for i in range(len(attributes)):
        for j in range(len(attributes[i])):
            for k in range(len(attributes)):
                if k == i:
                    no_discordance_table[i][k] = 0
                    continue
                if (attributes[i][j]*min_max[j] >= attributes[k][j]*min_max[j]):
                    if attributes[i][j]*min_max[j] - attributes[k][j]*min_max[j] > veto[j]:
                        no_discordance_table[k][i] = 0
    return no_discordance_table

def electre(concordance_table: np.ndarray, non_discordance_table,threshold: np.ndarray):
    links = np.zeros(concordance_table.shape)
    for i in range(len(concordance_table)):
        for j in range(len(concordance_table[i])):
            if concordance_table[i][j] >= threshold and non_discordance_table[i][j] == 1.0:
                links[i][j] = 1
    #Check for loops
    for i in range(len(links)):
        for j in range(i):
            if i!=j and links[i][j] == 1 and links[j][i] == 1:
                if concordance_table[i][j] > concordance_table[j][i]:
                    links[j][i] = 0
                elif concordance_table[i][j] < concordance_table[j][i]:
                    links[i][j] = 0
                elif concordance_table[i][j] == concordance_table[j][i]:
                    links[i][j] = 0
                    links[j][i] = 0
    return links

def get_core(links):
    core = []
    for i in range(len(links)):
        count = 0
        for j in range(len(links)):
            count += links[j][i]
        if count == 0:
            core.append(i)
    return core

def print_dominance(links):
    for i in range(len(links)):
        temp = []
        for j in range(len(links)):
            if links[i][j] == 1:
                temp.append(j)
        print(f"{i} : {temp}")

def directed_graph(table):
    graph = nx.DiGraph()
    for i in range(len(table)):
        graph.add_node(i)
    for i in range(len(table)):
        for j in range(len(table[i])):
            if table[i][j] == 1:
                graph.add_edge(i,j)

    nx.draw(graph, with_labels = True)
    plt.show()


classes = ["A","B","C"]
attributes = [[4500,7,7,8],
              [4000,7,3,8],
              [4000,5,7,8],
              [3500,5,7,5],
              [3500,5,7,8],
              [3500,3,3,8],
              [2500,3,7,5],]
min_max = [-1,1,1,1]
weights = [0.5,0.3,0.1,0.1]

pref_table = concordance(attributes,min_max,weights)

print("Concordance : ")
print(pref_table)

veto = [750,3,3.5,3.5]
non_discordance_table = no_discordance(attributes,min_max,veto)

print("Non Discordance : ")
print(non_discordance_table)

print("Electre : ")
links = electre(pref_table,non_discordance_table,0.7)
print_dominance(links)

core = get_core(links)
print(f"Core : {core}")