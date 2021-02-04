import numpy as np
import collections
import itertools
from itertools import repeat
import copy
import random
import pickle
import time
import os
from matplotlib import pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from data_processing import *
from hmdbn import * 
from baum_welch import *
from probs_update import *
from visualization import *

# seed for reproducability
random.seed(0)

# suppress overflow warnings that are resolved later
import warnings
warnings.filterwarnings("ignore")

'''
Identify all possible parent emissions given state
Arguments:
    state [state]: current state
    ri [int]: possible emissions of genes - i.e. (0, 1)
Returns:
    combinations [list]: list of possible parent emissions for state
    chi_dict [dict]: tracking parent emissions with integer
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
        # if past 0.5 probability, add time
        for t in range(T):
            next_val = np.around(P[0,t], 0)
            if prev_val != next_val:
                times.append(t)
                prev_val = next_val
        times.append(T)

    transition_times = np.sort(np.unique(times))        # sorted transition times
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
def identify_states(node_parents, P_list, T, plot=False):
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

    # normalize P so that sum of all curves = 1 at every t
    P_denom = np.tile(np.sum(P, axis=0), (len(states), 1))
    P = P/P_denom
    if plot == True:
        plot_puntative_graphs(node_parents, P_list, states, P)
    return states, P, n_seg

'''
Identify possible states and parent emissions from stationary graph
Arguments:
    timeseries [dict]: observations corresponding to gene key
    hmdbn_i [hmdbn]: hmdbn corresponding to gene_i
    init_hmdbns [dict]: hmdbns with single parent node corresponding to parent gene key
    T [float]: length of observations
    ri [int]: possible emissions of genes - i.e. (0, 1)
Returns:
    states [list]: possible graphs for timeseries
    state_emiss [list]: corresponding parent emissions
    chi_dicts [list]: corresponding dictionary for tracking parent emissions
    P [float]: posterior distribution matrix
    n_seg [int]: number of segments (state transitions)
'''
def putative_hidden_graphs(timeseries, hmdbn_i, init_hmdbns, T, ri):
    P_list = []
    child_gene = hmdbn_i.gene
    node_parents = hmdbn_i.parents

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
    child_gene [str]: name of gene_i
    timeseries [dict]: observations corresponding to gene key
    ri [list]: possible emissions of genes - i.e. (0, 1)
    filepath [str]: path to saved hmdbns
Returns:
    init_hmdbns [dict]: hmdbns with single parent node corresponding to parent gene key
'''
def pre_initialization(child_gene, timeseries, ri, filepath):
    # calculate posterior for every possible parent gene 
    filename = filepath+child_gene+'.pkl'

    init_hmdbns = {}
    for p_gene in timeseries.keys():
        if p_gene != child_gene:
            try:
                hmdbn_ij = load_model('models/preinitialization/', hidden_state(child_gene, [p_gene]))
            except:
                hmdbn_ij = hmdbn(child_gene, [p_gene])
                hmdbn_ij = hidden_markov_EM(timeseries, hmdbn_ij, ri, initialization=True)
                hmdbn_ij.save_model('models/preinitialization/')
            
            init_hmdbns[p_gene] = hmdbn_ij
    
    return init_hmdbns

'''
Perform structural expectation maximization to update HMDBN_i parameters
Arguments:
    timeseries [dict]: observations corresponding to gene key
    hmdbn_i [hmdbn]: hmdbn corresponding to gene_i
    ri [list]: possible emissions of genes - i.e. (0, 1)
    init_hmdbns [optional, dict]: hmdbns with single parent node corresponding to parent gene key
    intialization [optional, bool]: true if called by pre_initialization 
    delta [optional, float]: tolerance for convergence
Returns:
    hmdbn_i [hmdbn]: updated hmdbn corresponding to gene_i
'''
def hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns=None, initialization=False, delta=1e-3):
    child_gene = hmdbn_i.gene
    child_obs = timeseries.get(child_gene)[1:]              # align with i-1 parent observation

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
    # if prev_theta_cond is not None:
    hmdbn_i.bwbic = calculate_bwbic(child_gene, timeseries, states, chi_dicts, prev_theta_cond, prev_P)
    trans_probs, emiss_probs, init_probs = prev_probs
    hmdbn_i.pi, hmdbn_i.A, hmdbn_i.E = dict(init_probs), dict(trans_probs), dict(emiss_probs)
    hmdbn_i.hid_states = [state.parents for state in states]
    hmdbn_i.theta, hmdbn_i.posterior = prev_theta_cond, prev_P

    # return updated HMDBN
    return hmdbn_i

    '''
Perform structural expectation maximization to find best graph structure and HMDBN_i 
Arguments:
    child_gene [str]: name of gene_i
    c_hmdbn [hmdbn]: current hmdbn for gene_i
    best_hmdbn [hmdbn]: best hmdbn for gene_i
    genes [list]: list of all genes of interest
    timeseries [dict]: observations corresponding to gene key
Returns:
    c_hmdbn [hmdbn]: current hmdbn for gene_i
    best_hmdbn [hmdbn]: best hmdbn for gene_i
    update_count [int]: number of updates to best_hmdbn
'''
def worker(child_gene, c_hmdbn, best_hmdbn, genes, timeseries):
    print('-> on gene: '+str(child_gene))

    # get shuffled list of other genes
    other_parents = copy.deepcopy(genes)
    other_parents.remove(child_gene)
    random.shuffle(other_parents)

    ri = np.unique([all_obs for all_obs in timeseries.values()])
    init_hmdbns = pre_initialization(child_gene, timeseries, ri, 'data/')
    starting_parents = c_hmdbn.parents
    starting_bwbic = best_hmdbn.bwbic
    best_bwbic = starting_bwbic
    hmdbn_i = copy.deepcopy(c_hmdbn)
    update_count = 0

    for parent in other_parents:
        # if parent is not currently in list of parents, add it and see if BWBIC increases
        if parent not in hmdbn_i.parents:
            hmdbn_i.parents.append(parent)

            if len(hmdbn_i.parents) == 1:
                hmdbn_i = copy.deepcopy(init_hmdbns[parent])
            else:
                hmdbn_i = hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns, initialization=False)
            
            # set this hmdbn as best hmdbn if BWBIC is higher
            if hmdbn_i.bwbic > best_bwbic and hmdbn_i.bwbic is not None:                    
                best_hmdbn = copy.deepcopy(hmdbn_i)
                best_bwbic = best_hmdbn.bwbic
                update_count += 1
            else: 
                hmdbn_i.parents.remove(parent)

        # if parent is in list of parents, remove it and see if BWBIC increases
        else:
            if len(hmdbn_i.parents) > 1: 
                hmdbn_i.parents.remove(parent)
                hmdbn_i = hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns, initialization=False)

                if hmdbn_i.bwbic > best_bwbic:                   
                    best_hmdbn = copy.deepcopy(hmdbn_i)
                    best_bwbic = best_hmdbn.bwbic
                    update_count += 1
                else: 
                    hmdbn_i.parents.append(parent)

            else: # if there is only one parent, swap out parent and see if BWBIC increases
                hmdbn_i.parents.remove(parent)
                swapped_gene = random.choice(other_parents)
                hmdbn_i.parents.append(swapped_gene)
                hmdbn_i = hidden_markov_EM(timeseries, hmdbn_i, ri, init_hmdbns, initialization=False)

                if hmdbn_i.bwbic > best_bwbic:                   
                    best_hmdbn = copy.deepcopy(hmdbn_i)
                    best_bwbic = best_hmdbn.bwbic
                    update_count += 1
                else: 
                    hmdbn_i.parents.append(parent)
                    hmdbn_i.parents.remove(swapped_gene)

    output = '-> done with '+str(child_gene)+ ' (' + str(update_count) +' updates) \n updated parents '+str(best_hmdbn.parents)+' from '+ str(starting_parents) + '\n updated bwbic '+ str(best_bwbic) + ' from '+str(starting_bwbic)
            
    print(output)
    return (hmdbn_i, best_hmdbn, update_count)

'''
Run structural expectation maximization by calling worker function in parallel
Arguments:
    timeseries [dict]: observations corresponding to gene key
    c_hmdbns [optional, list]: current hmdbns for all genes (to resume training)
    best_hmdbns [optional, list]: best hmdbns for all genes (to resume training)
Returns:
    best_hmdbns [list]: best hmdbns for all genes
'''    
def run_structural_EM(timeseries, best_hmdbns=None, c_hmdbns=None):
    genes = list(timeseries.keys())

    # if not using a saved model, initialize
    if best_hmdbns is None and c_hmdbns is None:
        c_hmdbns, best_hmdbns = [], []
        for gene in genes:
            c_hmdbns.append(hmdbn(gene))                                                                               # keep track of current hmdbns
            best_hmdbns.append(hmdbn(gene, parents='no parents', bwbic=calculate_bwbic(gene, timeseries)))             # keep track of best hmdbns (initialize bwbic score with no parents)

    convergence = False
    iters = 0

    while not convergence:
        # run worker function in parallel
        print('\n\033[1mCURRENT ITERATION: '+str(iters)+'\033[0m')
        with ProcessPoolExecutor() as executor:
            results = executor.map(worker, genes, c_hmdbns, best_hmdbns, repeat(genes), repeat(timeseries))
        
        c_hmdbns, best_hmdbns, update_count = [], [], 0
        for result in list(results):
            c_hmdbns.append(result[0])
            best_hmdbns.append(result[1])
            update_count += result[2]

        save_hmdbns(c_hmdbns, 'models/iteration_'+str(iters)+'/temp_hmdbns/')
        save_hmdbns(best_hmdbns, 'models/iteration_'+str(iters)+'/hmdbns/')

        # if no changes, end algorithm
        if update_count == 0:
            convergence = True

        iters += 1
            
    return best_hmdbns

if __name__ == "__main__":
    timeseries = get_dataset('small_drosophlia')
    best_hmdbns = run_structural_EM(timeseries)
    save_hmdbns(best_hmdbns, 'models/final_hmdbns/')
    plot_posteriors(load_hmdbns('models/final_hmdbns/'))