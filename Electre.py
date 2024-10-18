import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_concordance(attributes, min_max, weights):
    """
    Generates the concordance table for the problem.

    Args:
        attributes: The attributes on which the data is being valued
        min_max: Array of either 1 for max or -1 for min
        weights: Weight of each attribute

    Returns:
        concordance_table : The concordance table of the given values.
    """

    #Initialization of the table at the right size and fill it with zeros
    concordance_table = np.zeros((len(attributes),len(attributes)))

    #Loop to iterate through each entity
    for i in range(len(attributes)):
        #Loop to iterate through the different attributes
        for j in range(len(attributes[i])):
            #Loop to iterate through the other entities to compare entities
            for k in range(len(attributes)):
                #If we're comparing the same entity then we skip the analysis and set its concordance at 0
                if k == i:
                    concordance_table[i][k] = 0
                    continue
                #If the entity is the better one for this attribute then we give it the corresponding weight
                if (attributes[i][j]*min_max[j] >= attributes[k][j]*min_max[j]):
                    concordance_table[i][k] += weights[j]
    return concordance_table

def get_non_discordance(attributes, min_max, veto):
    """
    Generates the non discordance table for the problem.

    Args:
        attributes: The attributes on which the data is being valued
        min_max: Array of either 1 for max or -1 for min
        veto: veto threshold for each attribute

    Returns:
        non_discordance_table : The concordance table of the given values.
    """

    #Initialization of the non discordance table at the right shape and fill it with ones
    non_discordance_table = np.ones((len(attributes),len(attributes)))

    #Loop to iterate through each entity
    for i in range(len(attributes)):
        #Loop to iterate through the different attributes
        for j in range(len(attributes[i])):
            #Loop to iterate through the other entities to compare entities
            for k in range(len(attributes)):
                #If we're comparing the same entity then we skip the analysis and set its non discordance to 0
                if k == i:
                    non_discordance_table[i][k] = 0
                    continue
                #If the entity is the better one for this attribute then we check if it will be vetoed
                if (attributes[i][j]*min_max[j] >= attributes[k][j]*min_max[j]):
                    #If the difference between the two attributes is over the veto then we veto it by setting its value to 0 in the table
                    if attributes[i][j]*min_max[j] - attributes[k][j]*min_max[j] > veto[j]:
                        non_discordance_table[k][i] = 0
    return non_discordance_table

def apply_electre(concordance_table: np.ndarray, non_discordance_table : np.ndarray,threshold):
    """
    Generates the graph links table for the problem.

    Args:
        concordance_table (np.ndarray) : The concordance table for the problem
        non_discordance_table (np.ndarray) : The non dicordance table for the problem
        threshold : Threshold to surpass to qualify (value between 0 and 1)

    Returns:
        links : The graph links table for the problem.
    """
    #Initialization of the links table with all its values set to 0
    links = np.zeros(concordance_table.shape)

    #Iterate through the entities
    for i in range(len(concordance_table)):
        #Iterate through its scores compared to the other entities 
        for j in range(len(concordance_table[i])):
            #If it has a higher value than the threshold and there isn't a problem with the non discordance then we confirm a link
            if concordance_table[i][j] >= threshold and non_discordance_table[i][j] == 1.0:
                links[i][j] = 1
    #We reiterate through our newly created links table to check for cycles in the graph
    for i in range(len(links)):
        #we check in an upside down L shape for links
        for j in range(i):
            #If there is a cyclic link between two nodes then we will fix the cyclic link
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
    """
    Gets the core that can be extracted from the graph

    Args:
        links (np.ndarray) : The graph links table to process

    Returns:
        core : List of all attributes in the core
    """
    #Initialization of the core
    core = []

    #We iterate through the graph links table
    for i in range(len(links)):
        #We entity will be part of the core
        isCore = True
        for j in range(len(links)):
            #If a value in the entity's column is equal to 1 then it won't be int the core so we stop looking
            if links[j][i] == 1.0:
                isCore = False
                break
        #If the value is a part of the core, then we add it to the list of core values
        if isCore:
            core.append(i)
    return core

def print_dominance(links):
    """
    Prints the dominated entities according to the electre algorithm from its graph links table in an easily readable way

    Args:
        links (np.ndarray) : The graph links table to process
    """
    for i in range(len(links)):
        temp = []
        for j in range(len(links)):
            if links[i][j] == 1:
                temp.append(j)
        print(f"{i} : {temp}")

def make_directed_graph(links):
    """
    Creates a directed graph to display from the graph links table

    Args:
        links (np.ndarray) : The graph links table to process
    """
    graph = nx.DiGraph()
    for i in range(len(links)):
        graph.add_node(i)
    for i in range(len(links)):
        for j in range(len(links[i])):
            if links[i][j] == 1:
                graph.add_edge(i,j)

    nx.draw(graph, with_labels = True)
    plt.show()

attributes = [[4500,7,7,8],
              [4000,7,3,8],
              [4000,5,7,8],
              [3500,5,7,5],
              [3500,5,7,8],
              [3500,3,3,8],
              [2500,3,7,5],]
min_max = [-1,1,1,1]
weights = [0.5,0.3,0.1,0.1]

concordance_table = get_concordance(attributes,min_max,weights)

print("Concordance : ")
print(concordance_table)

veto = [750,3,3.5,3.5]
non_discordance_table = get_non_discordance(attributes,min_max,veto)

print("Non Discordance : ")
print(non_discordance_table)

print("Electre : ")
links = apply_electre(concordance_table,non_discordance_table,0.7)
print_dominance(links)

core = get_core(links)
print(f"Core : {core}")