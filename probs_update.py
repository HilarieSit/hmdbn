import numpy as np
from numpy import inf
import collections
import itertools
import random

from baum_welch import *

'''
Initialize probibility dictionaries corresponding to states
Arguments:
    state_emiss [list]: possible parent emissions
    ri [int]: possible emissions of genes - i.e. (0, 1)
    T [float]: length of observations
    n_seg [int]: number of segments (state transitions)
Returns:
    trans_probs, emiss_probs, init_probs [dicts]: transition, emission, and initial probabilities
'''
def initialize_prob_dicts(state_emiss, ri, T, n_seg):
    # collection.defaultdict(dict) for initializing dict of dicts   
    init_probs = collections.defaultdict(dict)
    trans_probs = collections.defaultdict(dict)   
    emiss_probs = collections.defaultdict(lambda: collections.defaultdict(dict)) 

    Ri = len(ri)
    n_configs = len(state_emiss)
    pi_init = np.log(1/n_configs)
    if n_seg is None:
        trans_prob_init = np.log(0.05)
        self_prob_init = np.log(0.95)
    else:
        trans_prob_init = np.log(1/10)
        self_prob_init = np.log(1-(n_seg/10))

    for state_id, combinations in enumerate(state_emiss):
        init_probs[state_id] = pi_init
        for state_id2, _ in enumerate(state_emiss):
            if state_id == state_id2:
                trans_probs[state_id][state_id2] = self_prob_init
            else:
                trans_probs[state_id][state_id2] = trans_prob_init

        # initialize emiss with zeros (fill in later with calculate_theta)
        for gene_emiss in range(Ri):
            for parent_emiss in combinations:
                emiss_probs[state_id][gene_emiss][str(parent_emiss)] = 0  
                    
    return trans_probs, emiss_probs, init_probs

'''
Update probability dictionaries to correspond to observations
Arguments:
    current_gene [str]: gene_i
    obs [float]: timeseries observation array for gene_i
    states [list]: possible states (hidden graphs)
    probs [tuple]: initial, emission, transition probability dicts
    F, B, P [float]: forward, backward, posterior probability
    f_likelihood [float]: log likelihood score
Returns:
    trans_probs, init_probs [dicts]: updated transition and initial probabilities
'''
def update_probs(current_gene, obs, states, probs, F, B, P, f_likelihood):
    current_obs, timeseries = obs
    trans_probs, emiss_probs, init_probs = probs
    n_states = len(states)
    T = len(current_obs)

    # calculate pi (init_probs)
    pi_num = F[:,0] 
    pi_denom = sumLogProbsFunc(np.hsplit(F[:,0], n_states))
    pi = pi_num - pi_denom
    for h in range(n_states): 
        init_probs[h] = pi[h]

    # calculate A (trans_probs)
    A_num = collections.defaultdict(dict)   
    for prev_q, _ in enumerate(states):
        for q, state2 in enumerate(states): 
            s2_parents = state2.parents
            for t in range(T):
                # figure out parent observations for emiss probs
                back_parent_obs = get_parent_obs(current_gene, timeseries, s2_parents, t)
                
                # calculate numerator 
                A_count = F[prev_q,t-1]+trans_probs[prev_q][q]+emiss_probs[q][current_obs[t]][back_parent_obs]+B[q,t]-f_likelihood
                
                if t == 0:
                    A_num[prev_q][q] = A_count
                else:
                    A_num[prev_q][q] = sumLogProbs(A_num[prev_q][q], A_count)

        # calculate denominator
        A_denom = []
        for q2 in range(len(states)):
            A_denom.append(A_num[prev_q][q2])
        A_denom = sumLogProbsFunc(A_denom)

        # calculate transition probability
        for q2 in range(len(states)):
            trans_probs[prev_q][q2] = A_num[prev_q][q2]-A_denom

    return init_probs, trans_probs

'''
Calculate theta and update emission probabilties
Arguments:
    current_gene [str]: gene_i
    obs [float]: timeseries observation array for gene_i
    states [list]: possible states (hidden graphs)
    probs [tuple]: initial, emission, transition probability dicts
    F, B, P [float]: forward, backward, posterior probability
    f_likelihood [float]: log likelihood score
Returns:
    trans_probs, init_probs [dicts]: updated transition and initial probabilities
'''
def calculate_theta(current_gene, obs, states, state_emiss, chi_dicts, emiss_probs, P):
    """ calculates conditional probabilities of X given each config """
    current_obs, timeseries = obs
    T = len(current_obs)
    Ri = len(np.unique(current_obs))

    theta_cond, bwbic_score = [], []
    for state_id, state in enumerate(states):
        combinations = state_emiss[state_id]
        chi_dict = chi_dicts[state_id]

        # figure out possible variations of parent set
        s_parents = state.parents
        Gi = len(chi_dict)                                  # number of discrete states of parents
        theta_num = np.zeros((Ri, Gi))

        for t in range(T): 
            # from current_val & parent_vals, identify where to put count in chi (knonecker delta)
            current_val = current_obs[t]
            parent_vals = get_parent_obs(current_gene, timeseries, s_parents, t)
            chi_index = chi_dict.get(str(parent_vals))

            # calculate theta (do all i,jk at once)
            theta_num[current_val, chi_index] += P[state_id, t]

        # print(theta_num)
        theta_num_sum = np.sum(theta_num, axis=0)
        theta_denom = np.tile(theta_num_sum, (Ri, 1))
        theta_matrix = np.log(theta_num)-np.log(theta_denom)
        theta_matrix[np.where(theta_denom == 0)] = np.log(0.5)

        theta_cond.append(theta_matrix)

        # fill in emiss_probs
        for gene_emiss in range(Ri):
            for parent_emiss in combinations:
                emiss_probs[state_id][gene_emiss][str(parent_emiss)] = theta_matrix[gene_emiss, chi_dict.get(str(parent_emiss))]-np.log(len(combinations))
                if emiss_probs[state_id][gene_emiss][str(parent_emiss)] == -inf:
                    emiss_probs[state_id][gene_emiss][str(parent_emiss)] = 1

        bwbic_first_term = np.zeros((Ri, Gi))
        for t in range(T): 
            # from current_val & parent_vals, identify where to put count in chi (knonecker delta)
            current_val = current_obs[t]
            parent_vals = get_parent_obs(current_gene, timeseries, s_parents, t)
            chi_index = chi_dict.get(str(parent_vals))
            bwbic_first_term[current_val, chi_index] += P[state_id, t] * theta_matrix[current_val, chi_index]
        
        bwbic_first_term = np.sum(bwbic_first_term)
        bwbic_second_term = (Gi/2)*(Ri-1)*np.log(np.sum(P[state_id,:]))
        bwbic = bwbic_first_term-bwbic_second_term
        bwbic_score.append(bwbic)
    print([state.parents for state in states])
    print(bwbic_score)
    bwbic_score = np.mean(bwbic_score)
    return theta_cond, emiss_probs, bwbic_score

# def calculate_bwbic(theta_cond, chi_dict, timeseries, config):
