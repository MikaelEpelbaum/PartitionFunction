# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pgmpy.inference import BeliefPropagation
import numpy as np
from pgmpy.readwrite import UAIReader
from pgmpy.factors import factor_product
import random
import time
import math

from pgmpy.models.MarkovNetwork import MarkovNetwork
import itertools
from collections import Counter
import copy
import matplotlib.pyplot as plt
import networkx as nx
import operator
from pgmpy.models import JunctionTree

GRID_file = "./grid4x4.uai"


"""A helper function. You are free to use."""
def numberOfScopesPerVariable(scopes):
    # Initialize a dictionary to store the counts
    counts = {}
    # Iterate over each scope
    for scope in scopes:
        # Iterate over each variable in the scope
        for variable in scope:
            # Increment the count for the variable
            if variable in counts:
                counts[variable] += 1
            else:
                counts[variable] = 1
    # Sort the counts dictionary based on the frequency of variables in the scopes
    sorted_counts = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_counts


"""
You need to implement this function. It receives as input a junction tree object,
and a threshold value. It should return a set of variables such that in the 
junction tree that results from jt by removing these variables (from all bags), 
the size of the largest bag is threshold.
The heuristic used to remove RVs from the junction tree is: repeatedly remove the RV that appears in
the largest number of bags in the junction tree.
"""
def getCutset(jt, threshold):
    # todo: when I remove a variable from the nodes if a node didn't have the variable at first I completely loose it. is't wrong the node should continue to exist
    X = []
    JT = copy.deepcopy(jt)
    current_largest_node_size = max([len(list(clique)) for clique in JT.nodes()])
    while current_largest_node_size >= threshold:
        print(JT.nodes)
        G = nx.Graph()
        G.add_nodes_from(JT.nodes)
        G.add_edges_from(JT.edges)
        # finding variable that appear the most in the nodes (cliques)
        merged = list(itertools.chain(*JT.nodes()))
        variables_appearences = dict(Counter(merged))
        most_occurent_variable = sorted(variables_appearences.items(), key=lambda item: item[1], reverse=True)[0][0]
        print(most_occurent_variable, variables_appearences[most_occurent_variable])
        # most_occurent_variable1 = max(variables_appearences.items(), key=operator.itemgetter(1))[0]
        X.append(most_occurent_variable)

        # remove from nodes the most_occurent_variable
        new_adjacent_dic = {}
        # building new adjacency list of JT without X.
        for node in JT.nodes:
            if most_occurent_variable in node:
                new_node = tuple(item for item in node if item != most_occurent_variable)
                # new_JT.add_node(new_node)
                for neighbor in JT.adj[node]:
                    lst = list(neighbor)
                    if most_occurent_variable in lst:
                        lst.remove(most_occurent_variable)
                    lst = tuple(lst)
                    if new_node not in new_adjacent_dic:
                        new_adjacent_dic[new_node] = [lst]
                    else:
                        new_adjacent_dic[new_node].append(lst)
        # creating new JT based on previous one
        JT = JunctionTree()
        for node in new_adjacent_dic:
            JT.add_node(node)
        # Add edges to new_JT
        for node, neighbors in new_adjacent_dic.items():
            for neighbor in neighbors:
                if (neighbor, node) not in JT.edges and set(neighbor).intersection(set(node)):
                    JT.add_edge(node, neighbor)

        current_largest_node_size = max([len(list(clique)) for clique in JT.nodes()])
    return X


"""
You are provided with this function. It receives as input a junction tree object, the MarkovNetwork model,
and an evidence dictionary. It computes the partition function with this evidence.
"""
def computePartitionFunctionWithEvidence(jt, model, evidence):
    reducedFactors=[]
    for factor in jt.factors:
        evidence_vars = []
        for var in factor.variables:
            if var in evidence:
                evidence_vars.append(var)
        if evidence_vars:
            reduce_vars = [(var, evidence[var]) for var in evidence_vars]
            new_factor= factor.reduce(reduce_vars, inplace=False)
            reducedFactors.append(new_factor)
        else:
            reducedFactors.append(factor.copy())


    totalfactor = factor_product(*[reducedFactors[i] for i in range(0, len(reducedFactors))])
    var_to_marg = (
            set(model.nodes()) - set(evidence.keys())
    )
    marg_prod = totalfactor.marginalize(var_to_marg, inplace=False)
    return marg_prod.values

"""This function implements the ComputePartitionFunction algorithm using the wCutset and GenerateSample functions"""
def computePartitionFunction(markovNetwork, w, N):
    """
            implements Algorithm ComputePartitionFunction.

            Parameters
            ----------
            markovNetwork : instance of UIaReader class
                Markov network or junction tree used to compute the partition function

            int : W
                denotes the bound on the largest cluster of the junction tree.

            int : N
                denotes the number of samples.
    """
    Z = 0
    T = MarkovNetwork.to_junction_tree(MN)
    x = getCutset(T, w)
    print(x)
    print("s")

        

"""This function implements the experiments where the sampling distribution is Q^{RB}"""
def ExperimentsDistributionQRB(path= GRID_file):
    pass

"""This function implements the experiments where the sampling distribution Q is uniform"""
def ExperimentsDistributionQUniform(path= GRID_file):
    pass






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    """Part 1"""
    reader = UAIReader(GRID_file)
    MN = reader.get_model()
    JT = MarkovNetwork.to_junction_tree(MN)
    w = max([len(list(clique)) for clique in JT.nodes()])
    computePartitionFunction(MN, 5, N=50)
    """Part 2"""
    #print("grid4x4 Experiments:")
    ExperimentsDistributionQRB(GRID_file)
    ExperimentsDistributionQUniform(GRID_file)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
