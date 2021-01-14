import numpy as np
import collections
import itertools
import copy
import random
import pickle
import time
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

    chi_dict = {}
    for chi_index, combination in enumerate(combinations):
        chi_dict[str(list(combination))] = chi_index
    return combinations, chi_dict

def identify_transitions(P_list, T):
    # calculate transition times for all parents
    times = []
    for P in P_list:
        # keep track of the previous parent expression (position 0)
        prev_val = np.around(P[0,0], 0)
        # if change past 0.5 probability
        for t in range(T):
            next_val = np.around(P[0,t], 0)
            if prev_val != next_val:
                times.append(t)
                prev_val = next_val
        times.append(T)

    transition_times = np.sort(np.unique(times))    
    n_seg = len(transition_times)                       # number of segments
    return transition_times, n_seg

def identify_states(node_parents, P_list, T):
    # identify transition points in timeseries
    transition_times, n_seg = identify_transitions(P_list, T)

    # identify possible states from segment values
    states, parent_genes = [], []

    # calculate segment values before t
    segments = np.zeros((len(P_list), n_seg))
    for nP, P in enumerate(P_list):
        for nt, t in enumerate(transition_times):
            segments[nP, nt] = np.around(P[0,t-1], 0)

    # get list of possible parents for each t get column indice for all t
    # FIRST GET POSSIBLE STATES 
    state_list = []
    for st, tt in enumerate(transition_times):
        state_ind = np.argwhere(segments[:, st] == 1.)
        state_list.append([node_parents[int(i)] for i in state_ind])
    print(state_list)
    
    # search for unique parents for P 
    states = [] 
    [states.append(x) for x in state_list if x not in states]

    P = np.zeros([len(states), T])
    pt = 0
    print(transition_times)
    for state, tt in zip(state_list, transition_times):
        p_ind = states.index(state)
        P[p_ind, pt:tt] = 1
        pt = tt

    print(P)
    
    # for st, tt in enumerate(transition_times):
    #     all_states = []
    #     state_ind = np.argwhere(segments[:, st] == 1.)
    #     print('ind', state_ind)
    #     if state_ind.size == 0: 
    #         all_states.append([])
    #     else: 
    #         for ind in state_ind[0,:]:
    #             all_states.append(node_parents[int(ind)])
    #     print('all_states', all_states)
        # match all states to node_parents
        # P needs to match states
        
    # for t in range(T):
    #     for state in all_states:
    #         P[state, t] 


    # print(all_states)



        # if not state_ind:
        #     P[state_ind, prev_tt:tt] = 1
        # figure out the column number for P from state_ind
        # print('state_ind', state_ind)
        # for i in state_ind:
        #     print(state_ind)
        #     states = node_parents[int(i)]
        # print(states)
        # print([p.gene for p in states])
            # print(states.parents)
        # genes = 0
        # print(states.parents)
        # parent_states.append(states)
        # parent_genes.append(genes)
    
    # print(parent_genes)
    # initialize P matrix
    # i = 0
    # P = np.zeros([len(node_parents),T])
    # break_point = transition_times[i]
    # for t in range(T):
    #     if not parent_states[i]:
    #         curr_state = len(node_parents)-1
    #     else:
    #         curr_state = node_parents.index(parent_states[i])
    #     P[curr_state, t] = 1
    #     if t == break_point-1:
    #         i+=1
    #         break_point = transition_times[i]

    # print(P)

    return all_states, P, n_seg

def putative_hidden_graphs(timeseries, node_i, node_lib, init_posteriors, T, ri, no_parent_node):
    """ return list of graphs with all parent combinations, corresponding possible parent emissions & dict for tracking combinations """
    P_list = []
    child_gene = node_i.gene
    node_parents = node_i.parents

    # get the correct initial posteriors
    P_list = [init_posteriors[parent.gene] for parent in node_parents]

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

def pre_initialization(current_gene, node_lib, timeseries, filepath):
    # calculate posterior for every possible parent gene with child gene
    print('\n initializing')
    node_i = node_lib.get(current_gene)
    filename = filepath+current_gene+'.pkl'

    try:
        with open(filename, "rb") as fp:
            print('\n -> loading '+filename+' .........', end=" ", flush=True)
            init_posteriors = pickle.load(fp)
            print('complete')
    except:
        init_posteriors = {}
        for p_gene, p_node in node_lib.items():
            # if p_gene != current_gene:
            node_ij = node(current_gene, [p_node])
            P = structural_EM(timeseries, node_ij, node_lib, initialization=True)
            init_posteriors[p_gene] = P

        with open(filename, 'wb') as output:
            pickle.dump(init_posteriors, output)
        print('\n -> '+filename+' saved .........')
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

    no_parent_node = node(current_gene, [node_i])

    while not convergence:
        if not initialization:
            print('\n -> performing structural EM .........')
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

            print('parents: ', [parents.gene for parents in node_i.parents]) 

            # 3.1. identify putative hidden graphs
            configs, configs_combos, chi_dicts, P, n_seg = putative_hidden_graphs(timeseries, node_i, node_lib, init_posteriors, T, ri, no_parent_node)  

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

            # forward backward algorithm
            F, B, P, likelihood = forward_backward(obs, configs, probs)
            if likelihood - prev_likelihood < delta:
                q_convergence = True
            prev_likelihood = likelihood

        # plt.clf()
        # plt.plot(np.linspace(1, 65, 65), P[0,:], 'blue')
        # plt.plot(np.linspace(1, 65, 65), P[1,:], 'orange')
        # plt.title([parents.gene for parents in node_i.parents])
        # plt.show()

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
    node_lib = initialize_nodes(all_genes)
    current_gene = 'twi'
    print('\n\033[1mCURRENT GENE: '+current_gene+'\033[0m')

    # preinitialize 
    node_i, init_posteriors = pre_initialization(current_gene, node_lib, timeseries, 'data/')
    hmdbn = structural_EM(timeseries, node_i, node_lib, init_posteriors, initialization=False)






    # construct list of all nodes (corresponding to genes) & position dict

    gene = 'up'
    
    # perform structural EM on every gene

    hmdbn = structural_EM(obs, node_i, all_nodes, T, ri)
