import numpy as np
import collections
import itertools
import copy
import random
import pickle
import time
import os
import multiprocessing
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from data_processing import load_data
from hmdbn import * 
from baum_welch import *
from probs_update import *
from visualization import *

'''
Create a node library with genes of interest
Arguments:
    genes [list]: all genes of interest
Returns:
    node_lib [dict]: nodes corresponding to gene key
'''
def initialize_hmdbns(genes):
    hmdbn_lib = {}
    for gene in genes:
        hmdbn_lib[gene] = hmdbn(gene)
    return hmdbn_lib

def initialize_hmdbns2(genes):
    hmdbn_list = []
    for gene in genes:
        hmdbn_list.append(hmdbn(gene))
    return hmdbn_list

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
def putative_hidden_graphs(timeseries, node_i, init_hmdbns, T, ri):
    P_list = []
    child_gene = node_i.gene
    node_parents = node_i.parents

    # get the correct initial posteriors
    P_list = [init_hmdbns[parent].posterior for parent in node_parents]

    # identify most probable hidden states using P
    states, state_emiss, chi_dicts = [], [], []
    possible_parents, P, n_seg = identify_states(node_parents, P_list, T)

    for pparents in possible_parents:
        network = hidden_state(child_gene, pparents)
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
def pre_initialization(current_gene, timeseries, ri, filepath):
    # calculate posterior for every possible parent gene 
    filename = filepath+current_gene+'.pkl'

    init_hmdbns = {}
    for p_gene in timeseries.keys():
        if p_gene != current_gene:
            try:
                hmdbn_ij = load_model('preinit/', hidden_state(current_gene, [p_gene]))
            except:
                hmdbn_ij = hmdbn(current_gene, [p_gene])
                hmdbn_ij = hidden_markov_EM(timeseries, hmdbn_ij, ri, initialization=True)
                hmdbn_ij.save_model('preinit/')
            
            init_hmdbns[p_gene] = hmdbn_ij
    
    return init_hmdbns

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
def hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns=None, initialization=False, delta=1e-3):
    child_gene = hmdbn_i.gene
    child_obs = timeseries.get(child_gene)[1:]

    T = len(child_obs)
    obs = (child_obs, timeseries)

    ### identify putative hidden graphs from stationary network and initialize posterior
    if not initialization:
        states, state_emiss, chi_dicts, P, n_seg = putative_hidden_graphs(timeseries, hmdbn_i, init_hmdbns, T, ri)  

    else:
        # putative hidden graphs during initialization is no parent node & single parent node
        no_parent_node = hidden_state(child_gene, [])
        one_parent_node = hidden_state(child_gene, hmdbn_i.parents)
        states = [one_parent_node, no_parent_node]
        state_emiss, chi_dicts = [], []
        for state in states:
            combinations, chi_dict = identify_parent_emissions(state, ri)
            state_emiss.append(combinations)
            chi_dicts.append(chi_dict)
        # initialize posterior as 50-50
        P = np.ones([2, T])/2
        n_seg = 1

    ### set initial values for P(q|x,HMDBN), A, pi, theta, E
    if len(state_emiss) == 0:
        print(hmdbn_i.gene)
        print(hmdbn_i.parents)
        print([state.parents for state in states])
    trans_probs, emiss_probs, init_probs = initialize_prob_dicts(state_emiss, ri, T, n_seg)
    _, emiss_probs = calculate_theta(child_gene, obs, states, state_emiss, chi_dicts, emiss_probs, P)
    probs = (trans_probs, emiss_probs, init_probs)
    F, B, P, f_likelihood = forward_backward(child_gene, obs, states, probs)

    ### iteratively re-estimate transition parameter to improve P(q)
    q_convergence = False
    prev_likelihood = np.NINF
    prev_probs, prev_theta_cond, prev_P = None, None, None

    while q_convergence is False:
        # calculate probability of state h given x & HMDBN
        init_probs, trans_probs = update_probs(child_gene, obs, states, probs, F, B, P, f_likelihood)
        theta_cond, emiss_probs = calculate_theta(child_gene, obs, states, state_emiss, chi_dicts, emiss_probs, P)
        probs = (trans_probs, emiss_probs, init_probs)

        # baum welch algorithm
        F, B, P, likelihood = forward_backward(child_gene, obs, states, probs)

        # if increase in likelihood < delta, end loop
        if likelihood - prev_likelihood <= delta or np.isnan(likelihood):
            q_convergence = True

        else: 
            prev_likelihood = likelihood
            prev_probs, prev_theta_cond, prev_P = probs, theta_cond, P
    
    # update the hmdbn parameters
    if prev_theta_cond is not None:
        hmdbn_i.bwbic = calculate_bwbic(child_gene, timeseries, states, chi_dicts, prev_theta_cond, prev_P)
        trans_probs, emiss_probs, init_probs = prev_probs
        hmdbn_i.pi, hmdbn_i.A, hmdbn_i.E = dict(init_probs), dict(trans_probs), dict(emiss_probs)
    hmdbn_i.hid_states = [state.parents for state in states]
    hmdbn_i.theta, hmdbn_i.posterior = prev_theta_cond, prev_P

    # return updated HMDBN
    return hmdbn_i

def worker(child_gene, c_hmdbn, best_hmdbn):
    print('-> on gene: '+str(child_gene))
    ri = np.unique([all_obs for all_obs in timeseries.values()])
    init_hmdbns = pre_initialization(child_gene, timeseries, ri, 'data/')
    starting_parents = c_hmdbn.parents
    starting_bwbic = best_hmdbn.bwbic
    best_bwbic = starting_bwbic
    hmdbn_i = copy.deepcopy(c_hmdbn)
    update_count = 0
    
    for parent in all_genes:
        if child_gene != parent:
            if parent not in hmdbn_i.parents:
                hmdbn_i.parents.append(parent)

                if len(hmdbn_i.parents) == 1:
                    hmdbn_i = copy.deepcopy(init_hmdbns[parent])
                else:
                    hmdbn_i = hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns, initialization=False)
                
                if hmdbn_i.bwbic > best_bwbic and hmdbn_i.bwbic is not None:                    
                    best_hmdbn = copy.deepcopy(hmdbn_i)
                    best_bwbic = best_hmdbn.bwbic
                    update_count += 1
                else: 
                    hmdbn_i.parents.remove(parent)

            else:
                if len(hmdbn_i.parents) > 1: 
                    hmdbn_i.parents.remove(parent)

                    if len(hmdbn_i.parents) == 1:
                        hmdbn_i = copy.deepcopy(init_hmdbns[hmdbn_i.parents[0]])
                    else:
                        hmdbn_i = hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns, initialization=False)

                    if hmdbn_i.bwbic > best_bwbic:                   
                        best_hmdbn = copy.deepcopy(hmdbn_i)
                        best_bwbic = best_hmdbn.bwbic
                        update_count += 1
                    else: 
                        hmdbn_i.parents.append(parent)

    output = '-> done with '+str(child_gene)+ ' (' + str(update_count) +' updates) \n updated parents '+str(best_hmdbn.parents)+' from '+ str(starting_parents) + '\n updated bwbic '+ str(best_bwbic) + ' from '+str(starting_bwbic)
            
    print(output)
    return (hmdbn_i, best_hmdbn, update_count)
    

def run_structural_EM(timeseries, grn=None, temp_grn=None):
    genes = list(timeseries.keys())
    # if not using a saved model, initialize
    if grn is None and temp_grn is None:
        c_hmdbns, best_hmdbns = [], []
        for gene in genes:
            c_hmdbns.append(hmdbn(gene))                 # keep track of best hmdbn for each gene
            best_hmdbns.append(hmdbn(gene, bwbic=calculate_bwbic(gene, timeseries)))             # keep track of current hmdbn 

    convergence = False
    iters = 0

    while not convergence:
        print('\n\033[1mCURRENT ITERATION: '+str(iters)+'\033[0m')
        with ProcessPoolExecutor() as executor:
            results = executor.map(worker, genes, c_hmdbns, best_hmdbns)
        
        c_hmdbns, best_hmdbns, update_count = [], [], 0
        for result in list(results):
            c_hmdbns.append(result[0])
            best_hmdbns.append(result[1])
            update_count += result[2]

        save_grn2(c_hmdbns, 'models/iteration'+str(iters)+'/temp_grn/')
        save_grn2(best_hmdbns, 'models/iteration'+str(iters)+'/grn/')

        if update_count == 0:
            convergence = True

        iters += 1
            
    return best_hmdbns


if __name__ == "__main__":
    gene_id = {
    'eve': 12294,
    'gfl': 9244,
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
    best_hmdbns = run_structural_EM(timeseries)