import numpy as np
import collections
import itertools
import random

from data_processing import load_data
from baum_welch import *
from probs_update import *

class node:
    def __init__(self, gene, parents, theta_i):
        self.gene = gene                     # gene id
        self.parents = parents               # parent nodes/graphs
        self.theta_i = theta_i               # conditional probabilities of induced/repressed state

def possible_parent_emissions(config_parents, ri):
    combinations = [list(vals) for vals in itertools.product(ri, repeat=len(config_parents))]
    chi_dict = {}
    for chi_index, combination in enumerate(combinations):
        chi_dict[str(list(combination))] = chi_index
    return combinations, chi_dict

def possible_parent_states(node_parents, P_list):
    # identify transition points in timeseries
    trans_times = []
    for P in P_list:
        prev_val = None
        for t in range(T):
            if (prev_val and (prev_val != P[:,t])) or (t == T-1):
                trans_times.append(t)

    # identify posterior values
    segments = np.zeros([len(P_list), len(trans_times)])
    for h, P in enumerate(P_list):
        segments[h, :] = np.array([P[1,t-1] for t in trans_times])

    # identify possible states from segment values
    parent_states = []
    for t in range(len(trans_times)):
        indices = np.where(segments[:,t])
        parents = [node_parents[index].gene for index in indices]
        parent_states.append(parents)
    
    # initialize P
    i = 0
    P = np.zeros([len(parent_states),T])
    break_point = trans_times[0]
    for t in range(T):
        if t < breakpoint:
            current_state = parent_states[i]
            P[current_state, t] = 1
            break_point = trans_times[i+1]

    return parent_states, P

def putative_hidden_graphs(obs, node_i, all_nodes, G2INT):
    """ return list of graphs with all parent combinations, corresponding possible parent emissions & dict for tracking combinations """
    P_list = []
    node_parents = node_i.parents
    child_gene = node_i.gene

    # calculate posterior for every parent genes with child gene
    for parent in node_parents:
        init_network = node(child_gene, parent, theta_i=None)
        _, P = structural_EM(init_network, timeseries, all_nodes, G2INT, initialization=True)
        P_list.append(P)

    # identify most probable hidden states using P
    configs, configs_combos, chi_dicts = [], [], []
    possible_parents, P = possible_parent_states(node_parents, P_list)
    for parents in possible_parents:
        network = node(child_gene, parents, theta_i=None)
        parents_emiss, chi_dict = possible_parent_emissions(parents, ri)
        configs.append(network)
        configs_combos.append(parents_emiss)
        chi_dicts.append(chi_dict)

    return configs, configs_combos, chi_dicts, P

    # for config_parents in itertools.combinations(node_parents, r):
    #         new_network = node(gene, list(config_parents), theta_i=None)
    #         configs.append(new_network)

    #         # put combination of parent states in list corresponding to all_graphs
    #         all_parent_combos = [list(vals) for vals in itertools.product(ri, repeat=len(config_parents))]
    #         configs_combos.append(all_parent_combos)

    #         # chi dicts for corresponding combinations (# e.g. chi has four possibilities if one parent: 0-0, 0-1, 1-0, 1-1)
    #         chi_dict = {}
    #         for chi_index, combination in enumerate(all_parent_combos):
    #             chi_dict[str(list(combination))] = chi_index
    #         chi_dicts.append(chi_dict)

    # identify config parents from P


    # get powerset of parents
    # for r in range(0, len(node_parents)+1):
    #     for config_parents in itertools.combinations(node_parents, r):
    #         new_network = node(gene, list(config_parents), theta_i=None)
    #         configs.append(new_network)

    #         # put combination of parent states in list corresponding to all_graphs
    #         all_parent_combos = [list(vals) for vals in itertools.product(ri, repeat=len(config_parents))]
    #         configs_combos.append(all_parent_combos)

    #         # chi dicts for corresponding combinations (# e.g. chi has four possibilities if one parent: 0-0, 0-1, 1-0, 1-1)
    #         chi_dict = {}
    #         for chi_index, combination in enumerate(all_parent_combos):
    #             chi_dict[str(list(combination))] = chi_index
    #         chi_dicts.append(chi_dict)

def structural_EM(obs, node_i, all_nodes, G2INT, initialization=False):
    """ return HMDBN for gene """
    while not convergence:
        if not initialization:
            # 2. randomly change parents by adding or deleting parent node - introduce stats from previous 
            parents = node_i.parents
            n_parents = len(parents)

            if bool(random.getrandbits(1)):
                other_nodes = all_nodes.copy()
                # remove itself and parents
                other_nodes.remove(node_i)
                for parent in parents:
                    other_nodes.remove(parent)
                # add random parent
                parent_gene = np.random.choice(other_nodes)          
                node_i.parents.append(parent_gene)
            else:
                if n_parents > 0:
                    node_i.parents.pop(np.random.randint(0, n_parents))        

            # 3.1. identify putative hidden graphs (note: configs can have different combinations of parents from above)
            configs, configs_combos, chi_dicts, P = putative_hidden_graphs(obs, node_i, all_nodes, G2INT)
        else:
            # initialization step with single parent 
            configs = node_i
            config_combos, chi_dicts = possible_parent_states(node_i.parents, ri)
            P = np.ones(2, T))/2

        # recurring_parents
        print([parents.gene for parents in node_i.parents])  

        # 3.2. set initial values for P(q|x,HMDBN), A, pi / calculate theta & E
        trans_probs, emiss_probs, init_probs = initialize_prob_dicts(configs_combos, Ri)
        _, emiss_probs, _ = calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P)
        probs = (trans_probs, emiss_probs, init_probs)
        
        F, B, P, _ = forward_backward(obs, configs, probs)

        # 3.3. iteratively re-estimate transition parameter to improve P(q)
        q_convergence = False
        prev_likelihood = 0

        while not q_convergence:
            # calculate probability of config h given x & HMDBN
            init_probs, trans_probs = update_probs(obs, configs, configs_combos, probs, F, B)
            theta_cond, emiss_probs, bwbic_score = calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P)
            probs = (trans_probs, emiss_probs, init_probs)

            # forward backward algorithm
            F, B, P, likelihood = forward_backward(obs, configs, probs)
            
            if likelihood - prev_likelihood < delta:
                q_convergence = True
            prev_likelihood = likelihood

        # print([parents.gene for parents in node_i.parents])
        print(bwbic_score)
        print('[=========================] converged ')

        if not initialization:
            overall_bwbic_score = np.sum(bwbic_score)
            print(overall_bwbic_score)
            bwbic_ind = np.argmax(bwbic_score)
        # node_i.parents = [parents for parents in configs[bwbic_ind].parents]

        # 3.4 Calculate the BWBIC score on converged P, theta
        # bwbic_score = calculate_bwbic(gene, timeseries, theta, P, probs)

        # # save HMDBN with best BWBIC score
        # if bwbic_score > high_score:
        #     trans_probs, _, init_probs = tuple(probs)
        #     G = configs
        #     A = trans_probs
        #     pi = init_probs
        #     theta = theta_cond
        #     # update best score
        #     best_bwbic_score = bwbic_score
                
    # return best HMDBN
    return (G, theta, pi, A), P


if __name__ == "__main__":
    genes = {
    'eve': 12294,
    'gfl/lmd': 9244,
    'twi': 12573,
    'mlc1': 10147,
    'mhc': 4693,
    'prm': 4385,
    'actn': 8237,
    'up': 6990,
    'myo61f': 2013,
    'msp300': 11654}

    timeseries = load_data(genes, 'data/testing')

    # construct list of all nodes (corresponding to genes) & position dict
    genes = timeseries.keys()
    all_nodes = []
    G2INT = {}
    for i, gene in enumerate(genes):
        all_nodes.append(node(gene, parents=[], theta_i=None))
        G2INT[gene] = i

    # perform structural EM on every gene
     current_obs = timeseries.get(gene)                  # timeseries for current gene
    T = len(current_obs)                                # length of timeseries
    ri = np.unique(current_obs)                         # possible emissions for gene
    Ri = len(ri)                                        # number of possible emissions for gene
    obs = (current_obs, timeseries)                     # group current_obs with timeseries dict (easier arg to pass)

    convergence = False
    best_bwbic_score = 0
    delta = 1e-4
    node_i = all_nodes[G2INT.get(gene)]

    hmdbn = structural_EM('mlc1', timeseries, all_nodes, G2INT)
