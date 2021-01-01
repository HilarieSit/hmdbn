import numpy as np
import collections
import itertools
import random

from baum_welch import *

def initialize_prob_dicts(configs_combos, Ri):
    """ initialize init/trans/emiss prob dicts corresponding to configs """
    # collection.defaultdict(dict) for initializing dict of dicts   
    init_probs = collections.defaultdict(dict)
    trans_probs = collections.defaultdict(dict)   
    emiss_probs = collections.defaultdict(lambda: collections.defaultdict(dict)) 

    pi_init = np.log(1/len(config_combos))
    trans_prob_init = np.log(1-T)
    self_prob_init = np.log(1-h/m)

    for config_id, combinations in enumerate(config_combos):
        init_probs[config_id] = pi_init
        for config_id2, _ in enumerate(config_combos):
            if config_id == config_id2:
                trans_probs[config_id][config_id2] = self_prob_init
            else:
                trans_probs[config_id][config_id2] = trans_prob_init
        # initialize emiss with zeros then fill in using calculate_theta
        for gene_emiss in range(Ri):
            for parent_emiss in combinations:
                emiss_probs[config_id][gene_emiss][str(parent_emiss)] = 0  
                    
    return trans_probs, emiss_probs, init_probs

def update_probs(obs, configs, configs_combos, probs, F, B):
    current_obs, timeseries = obs
    trans_probs, emiss_probs, init_probs = probs
    n_configs = len(configs)
    T = len(current_obs)

    # calculate pi (init_probs)
    pi_num = F[:,0] 
    pi_denom = sumLogProbsFunc(np.hsplit(pi_num, n_configs))
    pi = pi_num - pi_denom
    for h, config in enumerate(configs): 
        init_probs[config] = pi[h]

    # calculate A (trans_probs)
    for t in range(len(current_obs)-1):
        for q, config in enumerate(configs):
            A_denom = sumLogProbsFunc(np.hsplit(F[q,:], T))
            for next_q, config2 in enumerate(configs): 
                # figure out parent observations for emiss probs
                conf2_parents = config2.parents
                back_parent_obs = str([timeseries.get(parent.gene)[t+1] for parent in conf2_parents])

                 # calculate numerator 
                A_num = F[q,t]+trans_probs[q][next_q]+emiss_probs[next_q][current_obs[t+1]][back_parent_obs]+B[next_q,t+1]
                A = A_num - A_denom
                if t == 1:
                    trans_probs[config][config2] = A
                else:
                    trans_probs[config][config2] = sumLogProbs(trans_probs[q][next_q], A)

    return init_probs, trans_probs

def calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P):
    """ calculates conditional probabilities of X given each config """
    current_obs, timeseries = obs
    T = len(current_obs)
    Ri = len(np.unique(current_obs))

    theta_cond, bwbic_score = [], []
    for config_id, config in enumerate(configs):
        combinations = configs_combos[config_id]
        chi_dict = chi_dicts[config_id]

        # figure out possible variations of parent set
        conf_parents = config.parents
        all_parent_obs = [timeseries.get(parent.gene) for parent in conf_parents]
        
        if conf_parents:
            Gi = len(chi_dict)                                  # number of discrete states of parents
            theta_num = np.zeros((Ri, Gi))

            for t in range(T): 
                # from current_val & parent_vals, identify where to put count in chi (knonecker delta)
                current_val = current_obs[t]
                parent_vals = [parent_obs[t] for parent_obs in all_parent_obs]
                chi_index = chi_dict.get(str(parent_vals))

                # calculate theta (do all i,jk at once)
                theta_num[current_val, chi_index] += P[config_id, t]

            theta_num_sum = sumLogProbsFunc(list(np.vsplit(theta_num, Ri)))
            theta_denom = np.tile(theta_num_sum, (Ri, 1))
            theta_matrix = np.exp(theta_num - theta_denom)

            theta_sum = np.sum(theta_matrix)
            theta_cond.append(theta_matrix)

            # fill in emiss_probs
            for gene_emiss in range(Ri):
                for parent_emiss in combinations:
                    emiss_probs[config_id][gene_emiss][str(parent_emiss)] = theta_matrix[gene_emiss, chi_dict.get(str(parent_emiss))]/theta_sum

        else:
            Gi = 1
            chi_index = 0
            theta_num = np.zeros([Ri])
            for t in range(T): 
                current_val = current_obs[t]
                theta_num[current_val] += P[config_id, t]

            theta_matrix = np.expand_dims(theta_num/T, axis=1)
            theta_cond.append(theta_matrix)

            for gene_emiss in range(Ri):
                emiss_probs[0][gene_emiss]['[]'] = theta_matrix[gene_emiss]

        bwbic_first_term = np.zeros((Ri, Gi))
        for t in range(T): 
            # from current_val & parent_vals, identify where to put count in chi (knonecker delta)
            current_val = current_obs[t]
            parent_vals = [parent_obs[t] for parent_obs in all_parent_obs]
            chi_index = chi_dict.get(str(parent_vals))
            bwbic_first_term[current_val, chi_index] += P[config_id, t] * theta_matrix[current_val, chi_index]

        bwbic_first_term = np.sum(bwbic_first_term)
        bwbic_second_term = (Gi/2)*(Ri-1)*np.log(np.sum(P[config_id,:]))
        bwbic = bwbic_first_term-bwbic_second_term
        bwbic_score.append(bwbic)

    return theta_cond, emiss_probs, bwbic_score