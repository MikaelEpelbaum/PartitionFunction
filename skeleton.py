# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from pgmpy.inference import BeliefPropagation, VariableElimination
from pgmpy.readwrite import UAIReader
from pgmpy.factors import factor_product
import random
import time
import math
from statistics import mean, stdev

from pgmpy.models.MarkovNetwork import MarkovNetwork
import itertools
import copy
import pandas as pd

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
    variables_appearences = numberOfScopesPerVariable(JT.nodes)
    most_occurent_variable = variables_appearences.pop(0)[0]
    largest_cluster_size = max([len(item[0]) for item in JT.nodes.items()])
    bool = True

    while largest_cluster_size >= threshold or bool:
        X.append(most_occurent_variable)
        # removing edges which contain the the most frequent variable (the variable is present in both nodes that are connected to the edge.
        old_edges, new_edges = [], []
        for edge in JT.edges:
            if most_occurent_variable in edge[0] or most_occurent_variable in edge[1]:
                old_edges.append(edge)
                cleaned_edge = (tuple(var for var in edge[0] if var != most_occurent_variable), tuple(var for var in edge[1] if var != most_occurent_variable))
                if set(cleaned_edge[0]).intersection(cleaned_edge[1]) and cleaned_edge[0] != cleaned_edge[1]:
                    new_edges.append(cleaned_edge)

        JT.remove_edges_from(old_edges)
        # --------- ca me rajoute les nodes des edge que jai rajouter un surplus
        JT.add_edges_from(new_edges)

        # removing the most frequent var from the clique nodes that hold it (factors).
        nodes_to_remove = []
        new_nodes = []
        for node in JT.nodes:
            if most_occurent_variable in node:
                nodes_to_remove.append(node)
                new_nodes.append(tuple(var for var in node if var != most_occurent_variable))
        JT.remove_nodes_from(nodes_to_remove)
        JT.add_nodes_from(new_nodes)

        if len(variables_appearences) == 0:
            bool = False
        if len(variables_appearences) > 0:
            most_occurent_variable = variables_appearences.pop(0)[0]
        largest_cluster_size = max([len(item[0]) for item in JT.nodes.items()])
    return X, JT


def generate_sample(X):
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
            new_factor = factor.reduce(reduce_vars, inplace=False)
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
def computePartitionFunction(MN, w, N, distribution="QRB"):
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
    original_T = copy.deepcopy(T)
    X, T = getCutset(T, w)

    # Q
    Q = []
    if distribution == "uniform":
        Q = 1/2**len(X)

        # generate sample
        for i in range(N):
            x = generate_sample(X)
            part_x = computePartitionFunctionWithEvidence(T, MN, x)
            t_x = part_x / Q
            Z = Z + t_x
    elif distribution == "QRB":
        belief_propagation = BeliefPropagation(original_T)

        Q = belief_propagation.query(X)
        permutations = list(itertools.product([0, 1], repeat=len(X)))
        binary_permutations = [{X[i]: perm[i] for i in range(len(X))} for perm in permutations]

        Q_probas = {}
        for perm in binary_permutations:
            Q_probas[tuple(perm.items())] = Q.get_value(**perm)

        for i in range(N):
            x = generate_sample(X)
            part_x = computePartitionFunctionWithEvidence(original_T, MN, x)
            t_x = part_x / Q_probas[tuple(x.items())]
            Z = Z + t_x
    return Z / N



"""This function implements the experiments where the sampling distribution is Q^{RB}"""
def ExperimentsDistributionQRB(path= GRID_file):
    df = pd.DataFrame(columns=[50, 100, 1000, 5000], index=[1, 2, 3, 4, 5])
    reader = UAIReader(GRID_file)
    MN = reader.get_model()

    for N in [50, 100, 1000, 5000]:
        for w in [5, 4, 3, 2, 1]:
            E_values = []
            Time_values = []

            random.seed(random.random())
            jt = MarkovNetwork.to_junction_tree(MN)
            X, jt = getCutset(jt, w)
            inference = VariableElimination(MN)
            query = inference.query(X)
            Real_Z = query.values.sum()


            for i in range(10):
                start_time = time.time()
                cur_Z_uni = computePartitionFunction(MN, w, N, distribution="QRB")
                end_time = time.time()
                delta = end_time - start_time
                Time_values.append(delta)

                E_values.append(abs(math.log(cur_Z_uni) - math.log(Real_Z)) / math.log(Real_Z))

            E_avr = mean(E_values)
            E_std = stdev(E_values)
            Time_avg = mean(Time_values)
            Time_std = stdev(Time_values)

            df.loc[w][N] = {
                "Time": (Time_avg + Time_std, Time_avg - Time_std),
                "Error": (E_avr + E_std, E_avr - E_std)
            }
            df.to_csv('cur_df_QRB.csv')

"""This function implements the experiments where the sampling distribution Q is uniform"""
def ExperimentsDistributionQUniform(path= GRID_file):
    df = pd.DataFrame(columns=[50, 100, 1000, 5000], index=[1, 2, 3, 4, 5])
    reader = UAIReader(GRID_file)
    MN = reader.get_model()

    for N in [50, 100, 1000, 5000]:
        for w in [5, 4, 3, 2, 1]:
            E_values = []
            Time_values = []

            random.seed(random.random())
            jt = MarkovNetwork.to_junction_tree(MN)
            X, jt = getCutset(jt, w)
            inference = VariableElimination(MN)
            query = inference.query(X)
            Real_Z = query.values.sum()

            for i in range(10):
                start_time = time.time()
                cur_Z_uni = computePartitionFunction(MN, w, N, distribution="uniform")
                end_time = time.time()
                delta = end_time-start_time
                Time_values.append(delta)

                E_values.append(abs(math.log(cur_Z_uni)-math.log(Real_Z))/math.log(Real_Z))


            E_avr = mean(E_values)
            E_std = stdev(E_values)
            Time_avg = mean(Time_values)
            Time_std = stdev(Time_values)

            df.loc[w][N] = {
                "Time": (Time_avg + Time_std, Time_avg - Time_std),
                "Error": (E_avr + E_std, E_avr - E_std)
            }
            df.to_csv('cur_df_uniform.csv')
    return df








# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    """Part 1.1"""
    reader = UAIReader(GRID_file)
    MN = reader.get_model()
    JT = MarkovNetwork.to_junction_tree(MN)
    w = max([len(list(clique)) for clique in JT.nodes()])
    partitionFunctionResult_uniform = computePartitionFunction(MN, 5, 50, "uniform")


    """Part 2"""
    partitionFunctionResult_QRB = computePartitionFunction(MN, 5, 50)


    """Part 3"""
    print("grid4x4 Experiments:")
    ExperimentsDistributionQUniform(GRID_file)
    ExperimentsDistributionQRB(GRID_file)
