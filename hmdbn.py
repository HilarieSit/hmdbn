import numpy as np
import itertools
import random

class node:
    def __init__(self, gene, parents, theta_i):
        self.gene = gene
        self.parents = parents # parent nodes
        self.theta_i = theta_i

class hmdbn:
    def __init__(self, node, theta, pi, A):
        self.node = node
        self.theta = theta
        self.pi = pi
        self.A = A

    def calculate_q(self):
        pass

    def calculate_emission(self):
        pass
        
# class hidden_graph:
#     """ Make hidden DBN from list of nodes
#     """
#     def __init__(self, genes, edges, theta, config_id):
#         # list of connected nodes
#         self.graph = self.assemble_graph(genes, edges, theta)
#         self.config = config_id
    
#     def assemble_graph(self, genes, edges):
#         self.graph = [None]*len(genes)
#         for gene_id in genes:
#             self.graph[gene_id] = Node(gene_id, [], [])
#         for parent_id, child_id in edges.items():
#             self.graph[child_id].parents.append(self.graph[parent_id])
#             # calculate theta_i
#             self.graph[child_id].theta_i.append(parent)
        
#     def calculate_joint_prob(self):
#         pass

def putative_hidden_graphs(node_i, theta_i):
    all_graphs = []
    parents = node_i.parents
    gene = node_i.gene
    # get powerset of parents
    for r in range(0, len(parents)+1):
        for combination in itertools.combinations(parents, r):
            new_network = node(gene, combination, theta_i)
            all_graphs.append(node)
    return all_graphs

def calculate_theta(obs, trans_probs, emiss_probs, init_probs):
    F, likelihood_f, B, likelihood_b, R = forward_backward(obs, trans_probs, emiss_probs, init_probs)
    numerator = kd*F
    denominator = kd*
    return theta

genes = ['eve', 'gfl/lmd', 'twi', 'mlc1', 'sls', 'mhc', 'prm', 'actn', 'up', 'myo61f', 'msp300']
t = 64          # time in time series
########### M STEP 
# 1. initial stationary network based on theta_i (probability connection with gene_j)
graph = []
G2INT = {}
for i, gene in enumerate(genes):
    graph.append(node(gene, parents=[], theta_i=[]))
    G2INT[gene] = i

# write this in parallel
for gene in genes:
    convergence = False
    while not convergence:
        # 2. Change parents - add or delete parent node 
        node_i = graph[G2INT.get(gene)]
        if bool(random.getrandbits(1)):
            parent_gene = np.random.choice(genes)
            node_i.parents.append(graph[G2INT.get(parent_gene)])                 # add random parent
        else:
            n_parents = len(node_i.parents)                                 # current # of parents 
            if n_parents > 0:
                node_i.parents.pop(np.random.randint(n_parents))     # remove random parent

        # 3. Transform stationary network to HMDBN_i
        # 3.1 identify putative hidden graphs
        putative_graphs = putative_hidden_graphs(node_i, theta_i=[])

        # 3.2 set initial values for pi, A and P(q|x,HMDBN) and furthermore, estimate theta
        h = len(putative_graphs)        # number of configs
        P = (1/h)*np.ones(h)            # probability of config h given x & HMDBN
        A = (1/h)*np.ones((h,h))    
        E = (1/h)*np.ones((h,t))    
        pi = (1/h)*np.ones(h)           # initial probability of config for X1 
        theta = calculate_theta(obs, A, E, pi) # follow equation 20
        q = # hidden graph sequence

        # 3.3 iteratively re-estimate transition parameter to improve P(q)
        for i in range(10):
            pi = # equation 6
            A = 
            pi = 
            theta = 

        # 3.4 Calculate the BWBIC score

