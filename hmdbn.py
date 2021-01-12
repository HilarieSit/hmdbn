import numpy as np
import collections
import itertools
import random
from matplotlib import pyplot as plt

from data_processing import load_data
from baum_welch import *
from probs_update import *

class node:
    def __init__(self, gene, parents):
        self.gene = gene                     # gene id
        self.parents = parents               # parent nodes/graphs

def initialize_nodes(genes):
    node_lib = {}
    for gene in genes:
        node_lib[gene] = node(gene, parents=[])
    return node_lib

def identify_parent_emissions(config, ri):
    config_parents = config.parents
    combinations = [list(vals) for vals in itertools.product(ri, repeat=len(config_parents))]
    # print(combinations)
    # print(ri)
    chi_dict = {}
    for chi_index, combination in enumerate(combinations):
        chi_dict[str(list(combination))] = chi_index
    return combinations, chi_dict

def identify_transitions(P_list, T):
    # collect list of transition times & end time
    transition_times = []
    for P in P_list:
        # keep track of the previous parent expression (position 0)
        prev_val = np.around(P[0,0], 0)
        print(prev_val)
        # if change past 0.5 probability
        for t in range(T):
            if (prev_val and (prev_val != P[:,t])) or (t == T-1):
                transition_times.append(t)
    # number of segments
    n_seg = len(transition_times)
    return transition_times, n_seg

def identify_states(node_parents, P_list, T):
    # identify transition points in timeseries
    transition_times, n_seg = identify_transitions(P_list, T)

    # figure out segment value
    segments = np.zeros([len(P_list), n_seg])
    for h, P in enumerate(P_list):
        segments[h, :] = np.array([P[0,t-1] for t in transition_times])

    # identify possible states from segment values
    parent_states = []
    for t in range(n_seg):
        indices = np.where(segments[:,t])[0].tolist()
        parents = [node_parents[index].gene for index in indices]
        parent_states.append(parents)
    
    # initialize P
    i = 0
    P = np.zeros([len(parent_states),T])
    break_point = transition_times[0]
    print(transition_times)
    for t in range(T):
        if t < break_point:
            current_state = indices[i]
            P[current_state, t] = 1
            break_point = transition_times[i+1]
            i+=1

    return parent_states, P, n_seg

def putative_hidden_graphs(timeseries, node_i, node_lib, init_posteriors, T, ri):
    """ return list of graphs with all parent combinations, corresponding possible parent emissions & dict for tracking combinations """
    P_list = []
    node_parents = node_i.parents
    child_gene = node_i.gene

    # get the correct initial posteriors
    P_list = [init_posteriors.get(parent) for parent in node_parents]

    # identify most probable hidden states using P
    configs, configs_combos, chi_dicts = [], [], []
    possible_parents, P, n_seg = identify_states(node_parents, P_list, T)

    for parents in possible_parents:
        network = node(child_gene, parents)
        parents_emiss, chi_dict = identify_parent_emissions([parents], ri)
        configs.append(network)
        configs_combos.append(parents_emiss)
        chi_dicts.append(chi_dict)

    return configs, configs_combos, chi_dicts, P, n_seg

def pre_initialization(current_gene, node_lib, timeseries):
    # calculate posterior for every possible parent gene with child gene
    node_i = node_lib.get(current_gene)
    init_posteriors = {}
    for p_gene, p_node in node_lib.items():
        if p_gene != current_gene:
            node_ij = node(current_gene, [p_node])
            P = structural_EM(timeseries, node_ij, node_lib, initialization=True)
            init_posteriors[p_gene] = P
    return node_i, init_posteriors


def structural_EM(timeseries, node_i, node_lib, init_posteriors=None, initialization=False):
    """ return HMDBN for gene """
    child_gene = node_i.gene
    current_obs = timeseries.get(child_gene)[1:]
    obs = (current_obs, timeseries)
    T = len(current_obs)
    ri = np.unique([all_obs for all_obs in timeseries.values()])

    convergence = False
    best_bwbic_score = 0
    delta = 1e-5

    no_parent_node = node(child_gene, [node_i])

    while not convergence:
        if not initialization:
            # 2. randomly change parents by adding or deleting parent node 
            parents = node_i.parents
            n_parents = len(parents)

            if bool(random.getrandbits(1)) or n_parents < 2:
                other_nodes = list(node_lib.values())
                # remove itself and parents
                other_nodes.remove(node_i)
                for parent in parents:
                    other_nodes.remove(parent)
                # add random parent
                parent_gene = np.random.choice(other_nodes)          
                node_i.parents.append(parent_gene)
            else:
                node_i.parents.pop(np.random.randint(0, n_parents))  

            # 3.1. identify putative hidden graphs
            configs, configs_combos, chi_dicts, P, n_seg = putative_hidden_graphs(timeseries, node_i, node_lib, init_posteriors, T, ri)  

        else:
            # initialization step with single parent 
            configs = [node_i, no_parent_node]
            configs_combos, chi_dicts = [], []
            for config in configs:
                combinations, chi_dict = identify_parent_emissions(config, ri)
                configs_combos.append(combinations)
                chi_dicts.append(chi_dict)
            P = np.ones([2, T])/2
            n_seg = None
    
        print('parents: ', [parents.gene for parents in node_i.parents])
        # 3.2. set initial values for P(q|x,HMDBN), A, pi / calculate theta & E
        trans_probs, emiss_probs, init_probs = initialize_prob_dicts(configs_combos, ri, T, n_seg)
        _, emiss_probs, _ = calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P)
        probs = (trans_probs, emiss_probs, init_probs)
        F, B, P, f_likelihood = forward_backward(obs, configs, probs)

        # 3.3. iteratively re-estimate transition parameter to improve P(q)
        q_convergence = False
        prev_likelihood = np.NINF

        while q_convergence is False:
            # calculate probability of config h given x & HMDBN
            init_probs, trans_probs = update_probs(obs, configs, configs_combos, probs, F, B, P, f_likelihood)
            theta_cond, emiss_probs, bwbic_score = calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P)
            probs = (trans_probs, emiss_probs, init_probs)
            # print('second', emiss_probs)

            # forward backward algorithm
            F, B, P, likelihood = forward_backward(obs, configs, probs)
            if likelihood - prev_likelihood < delta:
                q_convergence = True
            prev_likelihood = likelihood

        plt.clf()
        plt.plot(np.linspace(1, 65, 65), P[0,:], 'blue')
        plt.plot(np.linspace(1, 65, 65), P[1,:], 'orange')
        plt.title([parents.gene for parents in node_i.parents])
        plt.show()
        if initialization:
            return P

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
    gene_id = {
    'eve': 12294,
    'gfl/lmd': 9244,
    'twi': 12573,
    'mlc1': 10147,
    'mhc': 4693,
    'prm': 4385,
    'actn': 8237,
    'up': 6990,
    # 'myo61f': 2013,
    'msp300': 11654}


    all_genes = list(gene_id.keys())
    timeseries = load_data(gene_id, 'data/testing')
    print(timeseries)
    node_lib = initialize_nodes(all_genes)
    current_gene = 'twi'

    # preinitialize 
    node_i, init_posteriors = pre_initialization(current_gene, node_lib, timeseries)
    hmdbn = structural_EM(timeseries, node_i, node_lib, init_posteriors, initialization=False)






    # construct list of all nodes (corresponding to genes) & position dict

    gene = 'up'
    
    # perform structural EM on every gene

    hmdbn = structural_EM(obs, node_i, all_nodes, T, ri)
