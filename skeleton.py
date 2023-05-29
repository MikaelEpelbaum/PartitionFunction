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
    """
            calculate the cutset of the junction tree.
            at each iteration remove from the nodes the variable that appears the most in
            all of them till threshold.
            return the list of variables removed and a copy of the modified junction tree.

            Parameters
            ----------
            junction tree : jt
                the junction tree we are working with

            int : threshold
                threshold that has to be attained.

    """
    X = []
    JT = copy.deepcopy(jt)
    merged = list(itertools.chain(*[node.variables for node in JT.factors]))
    variables_appearences = dict(sorted(dict(Counter(merged)).items(), key=lambda item: item[1], reverse=True))
    most_occurent_variable = sorted(variables_appearences.items(), key=lambda item: item[1], reverse=True)[0][0]
    while variables_appearences[most_occurent_variable] >= threshold:
        X.append(most_occurent_variable)
        # removing edges which contain the the most frequent variable (the variable is present in both nodes that are connected to the edge.
        old_edges, new_edges = [], []
        for edge in JT.edges:
            if most_occurent_variable in edge[0] and most_occurent_variable in edge[1]:
                old_edges.append(edge)
                cleaned_edge = (tuple(var for var in edge[0] if var != most_occurent_variable), tuple(var for var in edge[1] if var != most_occurent_variable))
                if set(cleaned_edge[0]).intersection(cleaned_edge[1]):
                    new_edges.append(cleaned_edge)

        JT.remove_edges_from(old_edges)
        JT.add_edges_from(new_edges)

        # removing the most frequent var from the clique nodes that hold it (factors).
        for node in JT.factors:
            if most_occurent_variable in node.variables:
                node.variables.remove(most_occurent_variable)
        merged = list(itertools.chain(*[node.variables for node in JT.factors]))
        variables_appearences = dict(sorted(dict(Counter(merged)).items(), key=lambda item: item[1], reverse=True))
        most_occurent_variable = sorted(variables_appearences.items(), key=lambda item: item[1], reverse=True)[0][0]
    return X, JT


def generate_sample(X):
    print('a')
    temp = [random.randint(0, 1) for i in range(len(X))]
    x = {X[i] : temp[i] for i in range(len(X))}
    return x


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


from pgmpy.factors.discrete import TabularCPD
"""This function implements the ComputePartitionFunction algorithm using the wCutset and GenerateSample functions"""
def computePartitionFunction(markovNetwork, w, N, distribution="QRB"):
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
    # wCutset
    T = MarkovNetwork.to_junction_tree(MN)
    T_oren = copy.deepcopy(T)
    X, T = getCutset(T, w)
    X_oren, T_oren = getCutsetOren(T_oren, w)

    # Q
    Q = []
    if distribution == "uniform":
        Q = 1/2**len(X)
    elif distribution == "QRB":
        # BM = MarkovNetwork.to_bayesian_model(markovNetwork)
        #
        # # Get the factors from the Markov network
        # factors = markovNetwork.factors
        #
        # # Iterate over the factors
        # for factor in factors:
        #     # Get the variables and their cardinalities from the factor
        #     variables = factor.scope()
        #
        #     # Reshape the factor values to match the CPD shape
        #     values = factor.values
        #     if list(np.shape(factor.values)) == [2]:
        #         factor.values.reshape(-1)
        #
        #     # Create a CPD for the factor's variables in the BayesianModel
        #     cpd = TabularCPD(variables[0], 2, values) #evidence=variables[1:], evidence_card=[v for v in variables[1:]])
        #
        #     # Add CPD to the BayesianModel
        #     BM.add_cpds(cpd)
        #
        # belief_propagation = BeliefPropagation(BM)
        # evidence = generate_sample(X)
        evidence = generate_sample(X)
        belief_propagation = BeliefPropagation(markovNetwork)
        v = belief_propagation.query(variables=list(set(markovNetwork.nodes) - set(evidence.keys())), evidence=evidence)

        print(v)


    # generate sample
    for i in range(N):
        x = generate_sample(X)
        part_x = computePartitionFunctionWithEvidence(T, MN, x)
        t_x = part_x/Q
        Z = Z + t_x
    return Z/N



"""This function implements the experiments where the sampling distribution is Q^{RB}"""
def ExperimentsDistributionQRB(path= GRID_file):
    pass

"""This function implements the experiments where the sampling distribution Q is uniform"""
def ExperimentsDistributionQUniform(path= GRID_file):
    pass






# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # from pgmpy.factors.discrete import TabularCPD
    # from pgmpy.models import BayesianNetwork
    # from pgmpy.inference import BeliefPropagation
    # bayesian_model = BayesianNetwork([('A', 'J'), ('R', 'J'), ('J', 'Q'), ('J', 'L'), ('G', 'L')])
    # cpd_a = TabularCPD('A', 2, [[0.2], [0.8]])
    # cpd_r = TabularCPD('R', 2, [[0.4], [0.6]])
    # cpd_j = TabularCPD('J', 2, [[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]], ['R', 'A'], [2, 2])
    # cpd_q = TabularCPD('Q', 2, [[0.9, 0.2], [0.1, 0.8]], ['J'], [2])
    # cpd_l = TabularCPD('L', 2, [[0.9, 0.45, 0.8, 0.1], [0.1, 0.55, 0.2, 0.9]], ['G', 'J'], [2, 2])
    # cpd_g = TabularCPD('G', 2, [[0.6], [0.4]])
    # bayesian_model.add_cpds(cpd_a, cpd_r, cpd_j, cpd_q, cpd_l, cpd_g)
    # belief_propagation = BeliefPropagation(bayesian_model)
    # v = belief_propagation.query(variables=['J', 'Q'], evidence = {'A': 0, 'R': 0, 'G': 0, 'L': 1})
    # print(v)


    """Part 1.1"""
    reader = UAIReader(GRID_file)
    MN = reader.get_model()
    JT = MarkovNetwork.to_junction_tree(MN)
    w = max([len(list(clique)) for clique in JT.nodes()])
    partitionFunctionResult = computePartitionFunction(MN, 5, 50, "uniform")

    """Part 1.2"""
    partitionFunctionResult = computePartitionFunction(MN, 5, 50)
    print(partitionFunctionResult)
    """Part 2"""
    #print("grid4x4 Experiments:")
    ExperimentsDistributionQRB(GRID_file)
    ExperimentsDistributionQUniform(GRID_file)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
