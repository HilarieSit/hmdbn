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
from visualization import *

class node:
    def __init__(self, gene, parents):
        self.gene = gene                     # gene id
        self.parents = parents               # parent nodes/graphs

'''
Create a node library with genes of interest
Arguments:
    genes [list]: all genes of interest
Returns:
    node_lib [dict]: nodes corresponding to gene key
'''
def initialize_nodes(genes):
    node_lib = {}
    for gene in genes:
        node_lib[gene] = node(gene, parents=[])
    return node_lib

'''
Identify all possible parent emissions given state
Arguments:
    state [node]: node_i
    ri [int]: possible emissions of genes - i.e. (0, 1)
Returns:
    combinations [list]: list of possible parent emissions for state
    chi_dict [dict]: tracking parent emissions with int
'''
def identify_parent_emissions(state, ri):
    state_parents = state.parents
    # if not state_parents:
    #     # if '[]', parent is itself
    #     cp_length = 1
    # else:
    cp_length = len(state_parents)
    combinations = [list(vals) for vals in itertools.product(ri, repeat=cp_length)]

    chi_dict = {}
    for chi_index, combination in enumerate(combinations):
        chi_dict[str(list(combination))] = chi_index
    return combinations, chi_dict

'''
Calculate transition times from intialization posteriors
Arguments:
    P_list [list]: posterior distributions corresponding to node_parents
    T [float]: length of observations
Returns:
    transition_times [list]: transition indices 
    n_seg [int]: number of segments (state transitions)
'''
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

'''
Identify possible states from initalization posteriors and initialize posterior for HMDBN
Arguments:
    node_parents [list]: parents of stationary graph
    P_list [list]: posterior distributions corresponding to node_parents
    T [float]: length of observations
Returns:
    states [list]: possible graphs for timeseries
    P [float]: posterior distribution matrix
    n_seg [int]: number of segments (state transitions)
'''
def identify_states(node_parents, P_list, T):
    # identify transition points in timeseries
    transition_times, n_seg = identify_transitions(P_list, T)

    # calculate segment values before transition points and identify possible states
    states, parent_genes = [], []
    segments = np.zeros((len(P_list), n_seg))
    for nP, P in enumerate(P_list):
        for nt, tt in enumerate(transition_times):
            segments[nP, nt] = np.around(P[0, tt-1], 0)

    # get list of possible parents for each transition time, get column indices for all tt
    state_list = []
    for st, tt in enumerate(transition_times):
        state_ind = np.argwhere(segments[:, st] == 1.)
        state_list.append([node_parents[int(i)] for i in state_ind])
    
    # search for unique states
    states = [] 
    [states.append(x) for x in state_list if x not in states]
    
    # initialize P matrix
    P = np.ones([len(states), T])
    pt = 0
    print('node parents', node_parents)
    print(transition_times)
    for state_id, state in enumerate(states):
        for i, sP in enumerate(P_list):
            parent = node_parents[i]
            if parent in state:
                P[state_id, :] *= sP[0, :]
            else:
                P[state_id, :] *= sP[1, :]

        # normalize P so that it sums to 1
    P_denom = np.tile(np.sum(P, axis=0), (len(states), 1))
    P = P/P_denom
    # plot_puntative_graphs2(node_parents, P_list, states, P)
    return states, P, n_seg

'''
Identify possible states and parent emissions from stationary graph
Arguments:
    timeseries [dict]: observations corresponding to gene key
    node_i [node]: node corresponding to gene_i
    node_lib [dict]: nodes corresponding to gene key
    init_posteriors [optional, dict]: posterior distributions corresponding to parent gene key
    T [float]: length of observations
    ri [int]: possible emissions of genes - i.e. (0, 1)
Returns:
    states [list]: possible graphs for timeseries
    state_emiss [list]: corresponding parent emissions
    chi_dicts [list]: corresponding dictionary for tracking parent emissions
    P [float]: posterior distribution matrix
    n_seg [int]: number of segments (state transitions)
'''
def putative_hidden_graphs(timeseries, node_i, node_lib, init_posteriors, T, ri):
    P_list = []
    child_gene = node_i.gene
    node_parents = node_i.parents

    # get the correct initial posteriors
    P_list = [init_posteriors[parent] for parent in node_parents]

    # identify most probable hidden states using P
    states, state_emiss, chi_dicts = [], [], []
    possible_parents, P, n_seg = identify_states(node_parents, P_list, T)
    print('pp', possible_parents)

    for parents in possible_parents:
        network = node(child_gene, parents)
        parents_emiss, chi_dict = identify_parent_emissions(network, ri)
        states.append(network)
        state_emiss.append(parents_emiss)
        chi_dicts.append(chi_dict)

    return states, state_emiss, chi_dicts, P, n_seg

'''
Pre-initialization: calculate posteriors and BWBIC score for every possible parent gene of gene_i
Arguments:
    current_gene [str]: name of gene_i
    node_lib [dict]: nodes corresponding to gene key
    timeseries [dict]: observations corresponding to gene key
    filepath [str]: path to saved posteriors
Returns:
    node_i [node]: node corresponding to gene_i
    init_posteriors [dict]: posterior distributions corresponding to parent gene key
'''
def pre_initialization(current_gene, node_lib, timeseries, filepath):
    # calculate posterior for every possible parent gene 
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
        for p_gene in node_lib.keys():
            node_ij = node(current_gene, [p_gene])
            P, bwbic_score = structural_EM(timeseries, node_ij, node_lib, initialization=True)
            init_posteriors[p_gene] = P

        with open(filename, 'wb') as output:
            pickle.dump(init_posteriors, output)
        print('\n -> '+filename+' saved .........')
    return node_i, init_posteriors

'''
Perform structural expectation maximization on gene_i to find HMDBN_i
Arguments:
    timeseries [dict]: observations corresponding to gene key
    node_i [node]: node corresponding to gene_i
    node_lib [dict]: nodes corresponding to gene key
    init_posteriors [optional, dict]: posterior distributions corresponding to parent gene key
    intialization [optional, bool]: true if called by pre_initialization 
Returns:
    P [float]: posterior distribution matrix
    bwbic_score [float]: BWBIC score for final HMDBN_i
'''
def structural_EM(timeseries, node_i, node_lib, init_posteriors=None, initialization=False):
    child_gene = node_i.gene
    current_obs = timeseries.get(child_gene)[1:]
    # vals, counts = np.unique(current_obs, return_counts=True)
    # P = counts/np.sum(counts)
    # print(P)

    obs = (current_obs, timeseries)
    T = len(current_obs)
    ri = np.unique([all_obs for all_obs in timeseries.values()])

    convergence = False
    delta = 1e-3

    while not convergence:
        if not initialization:
            print('\n -> performing structural EM .........')
            # randomly change node parents by adding or deleting parent node 
            parents = node_i.parents
            n_parents = len(parents)

            if bool(random.getrandbits(1)) or n_parents < 2:
                other_nodes = list(node_lib.keys())
                other_nodes.remove(child_gene)
                for parent in parents:
                    other_nodes.remove(parent)
                # add random parent
                parent_gene = np.random.choice(other_nodes)          
                node_i.parents.append(parent_gene)
            else:
                node_i.parents.pop(np.random.randint(0, n_parents)) 

            # identify putative hidden graphs from stationary network
            states, state_emiss, chi_dicts, P, n_seg = putative_hidden_graphs(timeseries, node_i, node_lib, init_posteriors, T, ri)  
            print(states)

            if len(node_i.parents) == 1:
                n_seg = 1
                # return bwbic_score from pre-initialization

        else:
            # initialization step with single parent 
            no_parent_node = node(current_gene, [])
            states = [node_i, no_parent_node]
            state_emiss, chi_dicts = [], []
            for state in states:
                combinations, chi_dict = identify_parent_emissions(state, ri)
                state_emiss.append(combinations)
                chi_dicts.append(chi_dict)
            # initialize posterior as 50-50
            P = np.ones([2, T])/2
            n_seg = 1
    
        print('parents: ', [parents for parents in node_i.parents])

        # set initial values for P(q|x,HMDBN), A, pi, theta, E
        trans_probs, emiss_probs, init_probs = initialize_prob_dicts(state_emiss, ri, T, n_seg)
        _, emiss_probs, _ = calculate_theta(child_gene, obs, states, state_emiss, chi_dicts, emiss_probs, P)
        probs = (trans_probs, emiss_probs, init_probs)
        F, B, P, f_likelihood = forward_backward(child_gene, obs, states, probs)

        # iteratively re-estimate transition parameter to improve P(q)
        q_convergence = False
        prev_likelihood = np.NINF

        while q_convergence is False:
            # calculate probability of state h given x & HMDBN
            init_probs, trans_probs = update_probs(child_gene, obs, states, probs, F, B, P, f_likelihood)
            theta_cond, emiss_probs, bwbic_score = calculate_theta(child_gene, obs, states, state_emiss, chi_dicts, emiss_probs, P)
            probs = (trans_probs, emiss_probs, init_probs)

            # baum welch algorithm
            F, B, P, likelihood = forward_backward(child_gene, obs, states, probs)

            # if increase in likelihood < delta, end loop
            if likelihood - prev_likelihood < delta:
                q_convergence = True

            prev_likelihood = likelihood

        # if initialization is False:
        #     plot_posteriors(P, states)

        if initialization:
            return P, bwbic_score


        # print([parents.gene for parents in node_i.parents])
        print(bwbic_score)
        print('[=========================] converged ')

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
    'msp300': 11654}

    # load all data
    all_genes = list(gene_id.keys())
    timeseries = load_data(gene_id, 'data/testing')
    node_lib = initialize_nodes(all_genes)
    current_gene = 'up'
    print('\n\033[1mCURRENT GENE: '+current_gene+'\033[0m')

    # preinitialize 
    node_i, init_posteriors = pre_initialization(current_gene, node_lib, timeseries, 'data/')
    hmdbn = structural_EM(timeseries, node_i, node_lib, init_posteriors, initialization=False)



    # construct list of all nodes (corresponding to genes) & position dict

    gene = 'twi'
    
    # perform structural EM on every gene

    hmdbn = structural_EM(obs, node_i, all_nodes, T, ri)
