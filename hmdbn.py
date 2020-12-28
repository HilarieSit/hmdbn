import numpy as np
import collections
import itertools
import random
from forward_backward import *
from data_processing import load_data

class node:
    def __init__(self, gene, parents, theta_i):
        self.gene = gene                     # gene id
        self.parents = parents               # parent nodes/graphs
        self.theta_i = theta_i               # conditional probabilities of induced/repressed state

def putative_hidden_graphs(node_i, ri):
    """ return list of graphs with all parent combinations and observation combinations """
    all_graphs, combinations = [], []
    node_parents = node_i.parents
    gene = node_i.gene

    # get powerset of parents
    for r in range(0, len(node_parents)+1):
        for combination in itertools.combinations(node_parents, r):
            config_parents = list(combination)
            new_network = node(gene, config_parents, theta_i=None)
            all_graphs.append(new_network)

            # put combination of parent states in list corresponding to all_graphs
            combo_values = [list(vals) for vals in itertools.combinations(ri, len(config_parents))]
            combinations.append(combo_values)

    return all_graphs, combinations

def initialize_prob_dicts(config_combos, initialization_prob, timeseries, theta_cond):
    """ initialize init/trans/emiss prob dicts corresponding to configs """
    # collection.defaultdict(dict) for initializing dict of dicts   
    init_probs = collections.defaultdict(dict)
    trans_probs = collections.defaultdict(dict)   
    emiss_probs = collections.defaultdict(lambda: collections.defaultdict(dict)) 
    print('hi')
    for config_id, combinations in enumerate(config_combos):
        print(config_id)
        init_probs[config_id] = np.log(initialization_prob)
        for config_id2, _ in enumerate(config_combos):
            trans_probs[config_id][config_id2] = np.log(initialization_prob)
        # initialize emiss then fill in
        for gene_emiss in range(2):
            # print(gene_emiss)
            for parent_emiss in combinations:
                # print(parent_emiss)
                emiss_probs[config_id][gene_emiss][str(parent_emiss)] = 0  
    # print(emiss_probs)
    # emiss_probs = calculate_emiss_probs(config_combos, theta_cond, emiss_probs) 
    return trans_probs, emiss_probs, init_probs

def update_probs(probs, fb_output, configs, current_obs, timeseries):
    n_configs = len(configs)
    trans_probs, emiss_probs, init_probs = probs
    F, B, P = fb_output

    # calculate pi (init_probs)
    pi_num = F[0,:] 
    pi_denom = sumLogProbsFunc(np.hsplit(pi_num, n_configs))
    pi = pi_num - pi_denom
    for h, config in enumerate(configs): 
        init_probs[config] = pi[h]

    # calculate A (trans_probs)
    for t in range(1, len(current_obs)):
        for q, config in enumerate(configs):
            A_denom = sumLogProbsFunc(np.hsplit(F[q,t], n_configs))
            for next_q, config2 in enumerate(configs): 
                # calculate numerator 
                A_num = F[q,t]+trans_probs[config][config2]+emiss_probs[config2][current_obs[t+1]]+B[next_q,t+1]
                A = A_num - A_denom
                if t == 1:
                    trans_probs[config][config2] = A
                else:
                    trans_probs[config][config2] = sumLogProbs(trans_probs[config][config2], A)

    # calculate theta & emiss_probs
    theta_cond = calculate_theta(current_obs, timeseries, configs, P)
    emiss_probs = calculate_emiss_probs(configs, theta_cond, emiss_probs) 
    probs = (trans_probs, emiss_probs, init_probs)

    # forward backward algorithm
    fb_output, likelihood = forward_backward(current_obs, probs)
    return probs, theta_cond, fb_output, likelihood

def calculate_theta(current_obs, timeseries, configs_zip, P):
    """ calculates conditional probabilities of X given each config """
    T = len(current_obs)
    Ri = 2
    theta_cond = []
    for h, (config, combinations) in enumerate(configs_zip):
        # figure out possible variations of parent set
        conf_parents = config.parents
        all_parent_obs = [timeseries.get(parent.gene) for parent in conf_parents]
        
        if conf_parents:
            # e.g. chi has four possibilities if one parent: 0-0, 0-1, 1-0, 1-1
            chi_dict = {}
            for chi_index, combination in enumerate(combinations):
                chi_dict[str(list(combination))] = chi_index
            Gi = len(chi_dict)                                  # number of discrete states of parents
            theta_num = np.zeros((Ri, Gi))

            for t in range(len(current_obs)): 
                # from current_val & parent_vals, identify where to put count in chi (knonecker delta)
                current_val = current_obs[t]
                parent_vals = [parent_obs[t] for parent_obs in all_parent_obs]
                chi_index = chi_dict.get(str(parent_vals))

                # calculate theta (do all i,jk at once)
                theta_num[current_val, chi_index] += P[t,h]

            theta_num_sum = sumLogProbsFunc(list(np.vsplit(theta_num, Ri)))
            theta_denom = np.tile(theta_num_sum, (Ri, 1))
            # print(np.exp(theta_num - theta_denom))
            # print(config)
            theta_cond.append(np.exp(theta_num - theta_denom))
            
        else:
            # something is wrong here
            theta_array = np.zeros([Ri, T])
            for t in range(len(current_obs)): 
                current_val = current_obs[t]
                theta_array[current_val, t] = 1
            theta_cond.append(theta_array)

    return theta_cond, emiss_probs

def calculate_emiss_probs(config_combos, theta_cond, emiss_probs):
    # theta is a list of matrices
    for config_id, theta_matrix in enumerate(theta_cond):
        for gene_emiss in range(2):
            for parent_emiss in config_combos[config_id]:
                emiss_probs[config_id][gene_emiss][parent_emiss] = theta_cond[gene_emiss, chi_index.get(str(parent_emiss))]
    return emiss_probs

def structural_EM(gene, timeseries, all_nodes, G2INT):
    """ return HMDBN for gene """
    current_obs = timeseries.get(gene)
    ri = np.unique(current_obs)
    T = len(current_obs)
    convergence = False
    best_bwbic_score = 0
    delta = 1e-4

    while not convergence:
        # 2. randomly change parents by adding or deleting parent node 
        node_i = all_nodes[G2INT.get(gene)]
        n_parents = len(node_i.parents)
        other_nodes = all_nodes.copy()
        other_nodes.pop(G2INT.get(gene))
        if bool(random.getrandbits(1)):
            # add random parent
            parent_gene = np.random.choice(other_nodes)          
            node_i.parents.append(parent_gene)
        else:
            # remove random parent
            if n_parents > 0:
                node_i.parents.pop(np.random.randint(n_parents))          

        # 3.1. identify putative hidden graphs
        configs, config_combos = putative_hidden_graphs(node_i, ri)
        n_configs = len(configs)
        
        ### OKAY ###

        # 3.2. set initial values for P(q|x,HMDBN), A, pi / calculate theta & E
        initialization_prob = 1/n_configs
        P = initialization_prob*np.ones((T, n_configs))
        theta_cond = calculate_theta(current_obs, timeseries, zip(configs, config_combos), P)
        probs = initialize_prob_dicts(config_combos, initialization_prob, timeseries, theta_cond)
        if n_parents == 0:
            _, emiss_probs, _ = probs
            likelihood = 0
            # when there is only one possible state
            # for t in current_obs:
            #     likelihood += emiss_probs[0][t]
        else: 
            fb_output, _ = forward_backward(current_obs, probs)

            # 3.3. iteratively re-estimate transition parameter to improve P(q)
            q_convergence = False
            prev_likelihood = 0
            while not q_convergence:
                # calculate probability of config h given x & HMDBN
                probs, theta_cond, fb_output, likelihood = update_probs(probs, fb_output, configs, current_obs, timeseries)
                if likelihood - prev_likelihood < delta:
                    q_convergence = True
                prev_likelihood = likelihood

        # # 3.4 Calculate the BWBIC score
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
    return G, theta, pi, A, P


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
    for gene in genes:
        hmdbn = structural_EM(gene, timeseries, all_nodes, G2INT)
