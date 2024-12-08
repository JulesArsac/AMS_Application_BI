import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def generate_concordance_table(attributes, min_max, weights):
    """
    Generates the preference table for the problem.

    Args:
        attributes: The attributes on which the data is being valued
        min_max: Array of either 1 for max or -1 for min
        weights: Weight of each attribute

    Returns:
        pref_table : The preference table of the given values.
    """
    concordance_table = np.zeros((len(attributes),len(attributes)))
    for i in range(len(attributes)):
        for j in range(len(attributes[i])):
            for k in range(len(attributes)):
                if k == i:
                    concordance_table[i][k] = 0
                    continue
                if (attributes[i][j]*min_max[j] > attributes[k][j]*min_max[j]):
                    concordance_table[i][k] += weights[j]
    return concordance_table

def generate_concordance_table_threshold(attributes, min_max, weights, thresholds):
    concordance_table = np.zeros((len(attributes),len(attributes)))
    for i in range(len(attributes)):
        for j in range(len(attributes[i])):
            for k in range(len(attributes)):
                if k == i:
                    concordance_table[i][k] = 0
                    continue
                if (attributes[i][j]*min_max[j] > attributes[k][j]*min_max[j]):
                    if (np.abs(attributes[i][j] - attributes[k][j]) < thresholds[j]):
                        concordance_table[i][k] += weights[j] * (np.abs(attributes[i][j] - attributes[k][j]) / thresholds[j])
                    else:
                        concordance_table[i][k] += weights[j]
    return concordance_table

def make_directed_graph(save_path,links,labels = None):
    """
    Creates a directed graph to display from the graph links table

    Args:
        links (np.ndarray) : The graph links table to process
    """
    graph = nx.DiGraph()
    if labels is None:
        labels =  range(len(links))
    for i in labels:
        graph.add_node(i)
    for i in range(len(links)):
        for j in range(len(links[i])):
            if links[i][j] >= 1:
                graph.add_edge(labels[i],labels[j])

    nx.draw(graph, with_labels = True, pos=nx.spring_layout(graph, k=0.05, iterations=200),node_size = 6000)
    plt.savefig(save_path)
    plt.close()

def order_to_graph(order):
    links = np.zeros((len(order),len(order)))
    for i in range(len(order)-1):
        links[order[i]][order[i+1]] = 1
    return links

def promethee_I(attributes,min_max,weights,labels,save_fig,thresholds = None):
    if thresholds is None:
        concordance_table = generate_concordance_table(attributes,min_max,weights)
    else:
        concordance_table = generate_concordance_table_threshold(attributes,min_max,weights,thresholds)
    print("Concordance Table : ")
    print(concordance_table)
    phi_plus = np.array([sum(i) for i in concordance_table])
    print("Φ+ : ",phi_plus)
    phi_minus = np.array([sum(i) for i in concordance_table.T])
    print("Φ- : ",phi_minus)
    sort_index_plus = list(reversed(np.argsort(phi_plus)))
    links_plus = order_to_graph(sort_index_plus)
    print("Choice order Φ+ : ",[labels[i] for i in sort_index_plus])
    sort_index_minus = list(np.argsort(phi_minus))
    links_minus = order_to_graph(sort_index_minus)
    print("Choice order Φ+ : ",[labels[i] for i in sort_index_minus])
    links = links_plus+links_minus
    for i in range(len(links)):
        for j in range(len(links[i])):
            if links[i][j] >= 1 and links[j][i] >= 1:
                links[i][j] = 0
                links[j][i] = 0
    make_directed_graph(save_fig,links,labels)

def promethee_II(attributes,min_max,weights,labels,save_fig,thresholds = None):
    if thresholds is None:
        concordance_table = generate_concordance_table(attributes,min_max,weights)
    else:
        concordance_table = generate_concordance_table_threshold(attributes,min_max,weights,thresholds)
    print("Concordance Table : ")
    print(concordance_table)
    phi_plus = np.array([sum(i) for i in concordance_table])
    print("Φ+ : ",phi_plus)
    phis_minus = np.array([sum(i) for i in concordance_table.T])
    print("Φ- : ",phis_minus)
    phi = phi_plus - phis_minus
    print("Φ  : ",phi)
    sort_index = list(reversed(np.argsort(phi)))
    links = order_to_graph(sort_index)
    print("Choice order : ",[labels[i] for i in sort_index])
    make_directed_graph(save_fig,links,labels)

