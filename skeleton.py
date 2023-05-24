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
    X = []
    JT = copy.deepcopy(jt)
    merged = list(itertools.chain(*[node.variables for node in JT.factors]))
    variables_appearences = dict(Counter(merged))
    most_occurent_variable = sorted(variables_appearences.items(), key=lambda item: item[1], reverse=True)[0][0]
    while variables_appearences[most_occurent_variable] >= threshold:
        X.append(most_occurent_variable)
        for node in JT.factors:
            if most_occurent_variable in node.variables:
                node.variables.remove(most_occurent_variable)
        merged = list(itertools.chain(*[node.variables for node in JT.factors]))
        variables_appearences = dict(Counter(merged))
        most_occurent_variable = sorted(variables_appearences.items(), key=lambda item: item[1], reverse=True)[0][0]
    return X

def generate_sample(X, N):
    samples = [0] * len(X)
    for i in range(N):
        p = random.random()
        samples[math.floor(p * len(X))] += 1
    return [sample / N for sample in samples]


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
    X = getCutset(T, w)
    print(X)
    x = generate_sample(X, N)
    print(x)



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
