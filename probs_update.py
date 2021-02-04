import numpy as np
import collections

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

    # initialize transition probs
    trans_prob_init = np.log(1/T)
    self_prob_init = np.log(1-(n_seg/T))

    for state_id, combinations in enumerate(state_emiss):
        init_probs[state_id] = pi_init
        for state_id2, _ in enumerate(state_emiss):
            if state_id == state_id2:
                trans_probs[state_id][state_id2] = self_prob_init
            else:
                trans_probs[state_id][state_id2] = trans_prob_init

        # initialize emission probs with zeros (fill in later in calculate_theta)
        for gene_emiss in range(Ri):
            for parent_emiss in combinations:
                emiss_probs[state_id][gene_emiss][str(parent_emiss)] = 0  

    return trans_probs, emiss_probs, init_probs

'''
Update probability dictionaries to correspond to observations
Arguments:
    child_gene [str]: gene_i
    obs [float]: timeseries observation array for gene_i
    states [list]: possible states (hidden graphs)
    probs [tuple]: initial, emission, transition probability dicts
    F, B, P [float]: forward, backward, posterior probability
    f_likelihood [float]: log likelihood score
Returns:
    trans_probs, init_probs [dicts]: updated transition and initial probabilities
'''
def update_probs(child_gene, obs, states, probs, F, B, P, f_likelihood):
    child_obs, timeseries = obs
    trans_probs, emiss_probs, init_probs = probs
    n_states = len(states)
    T = len(child_obs)

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
                back_parent_obs = get_parent_obs(timeseries, s2_parents, t)
                
                # calculate numerator 
                A_count = F[prev_q,t-1]+trans_probs[prev_q][q]+emiss_probs[q][child_obs[t]][back_parent_obs]+B[q,t]-f_likelihood

                if t == 0:
                    A_num[prev_q][q] = A_count
                else:
                    # if value is small, don't bother adding it
                    if np.exp(A_count) > 1e-24:
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
    child_gene [str]: gene_i
    obs [float]: timeseries observation array for gene_i
    states [list]: possible states (hidden graphs)
    state_emiss [list]: possible parent emissions corresponding to states
    chi_dicts [list]: dictionaries for tracking parent emissions corresponding to states
    emiss_probs [dict]: emission probabilities
    P [float]: posterior probability
Returns:
    theta_cond [list]: theta matrices
    emiss_probs [dict]: updated emission probabilities
'''
def calculate_theta(child_gene, obs, states, state_emiss, chi_dicts, emiss_probs, P):
    child_obs, timeseries = obs
    T = len(child_obs)
    ri, counts = np.unique(child_obs, return_counts=True)
    Ri = len(ri)

    # emission probability given X_i
    probs = counts/np.sum(counts)

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
            current_val = child_obs[t]
            parent_vals = get_parent_obs(timeseries, s_parents, t)
            chi_index = chi_dict.get(str(parent_vals))

            # calculate theta (do all i,jk at once)
            theta_num[current_val, chi_index] += P[state_id, t]

        theta_num_sum = np.sum(theta_num, axis=0)
        theta_denom = np.tile(theta_num_sum, (Ri, 1))
        theta_denom[np.where(theta_denom == 0)] = 1                 # if parent emission state is not seen in data, put in number to prevent division by zero
        theta_matrix = np.log(theta_num)-np.log(theta_denom)

        theta_cond.append(theta_matrix)

        # fill in emiss_probs
        for gene_emiss in range(Ri):
            for parent_emiss in combinations:
                emiss_probs[state_id][gene_emiss][str(parent_emiss)] = theta_matrix[gene_emiss, chi_dict.get(str(parent_emiss))]+np.log(probs[gene_emiss])

    return theta_cond, emiss_probs

'''
Calculate BWBIC scores for HMDBN
Arguments:
    child_gene [str]: gene_i
    timeseries [dict]: observations corresponding to gene key
    states [optional, list]: possible states (hidden graphs); if not provided, calculate starting BWBIC
    chi_dicts [optional, list]: dictionaries for tracking parent emissions corresponding to states
    theta [optional, float]: theta matrix
    P [optional, float]: posterior probability
Returns:
    bwbic_score [float]: single score for hmdbn
'''
def calculate_bwbic(child_gene, timeseries, states=None, chi_dicts=None, thetas=None, P=None):
    child_obs = timeseries.get(child_gene)[1:]
    T = len(child_obs)
    ri, counts = np.unique(child_obs, return_counts=True)
    Ri = len(ri)

    if states is not None:
        bwbic_score = []
        for state_id, state in enumerate(states):
            chi_dict = chi_dicts[state_id]
            s_parents = state.parents
            Gi = len(chi_dict)                                  # number of discrete states of parents
            theta_matrix = thetas[state_id]

            bwbic_first_term = 0
            for t in range(T): 
                # from current_val & parent_vals, identify where to put count in chi (knonecker delta)
                current_val = child_obs[t]
                parent_vals = get_parent_obs(timeseries, s_parents, t)
                chi_index = chi_dict.get(str(parent_vals))
                bwbic_first_term += P[state_id, t] * theta_matrix[current_val, chi_index]
            
            bwbic_second_term = (Gi/2)*(Ri-1)*np.log(np.sum(P[state_id,:]))
            bwbic = bwbic_first_term-bwbic_second_term
            bwbic_score.append(bwbic)
        bwbic_score = np.sum(bwbic_score)

    else:
        # 0.5 is estimated, but close enough since we took median
        bwbic_score = .5*Ri*T*np.log(.5)-.5*np.log(.5*T)

    return bwbic_score