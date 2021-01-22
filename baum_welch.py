import numpy as np

'''
Calculate sum of log probabilities for two values/arrays
Arguments:
    a, b: log probabilities to sum
Returns:
	sum of log probabilities
'''
def sumLogProbs(a, b):
    " function for calculating sumLogProbs for two values/arrays (vectorized)"
    b_a = np.expand_dims(a+np.log(1+np.exp(b-a)), axis=0)
    a_b = np.expand_dims(b+np.log(1+np.exp(a-b)), axis=0)
    return np.squeeze(np.amax(np.concatenate((b_a, a_b), axis=0), axis=0))

'''
Calculate sum of log probabilities for list of values/arrays
Arguments:
    args: list of log probabilities to sum
Returns:
	sum of log probabilities
'''
def sumLogProbsFunc(args):
    " function for calculating sumLogProbs of list of values/arrays"
    if len(args) == 1:
        return args[0]
    else:
        sumtotal = sumLogProbs(args[0], args[1])
        for term in args[2:]:
            sumtotal = sumLogProbs(sumtotal, term)
        return sumtotal

'''
Return observations for all parents at specified time 
Arguments:
    current_gene [str]: gene of interest 
    timeseries [dict]: observations corresponding to gene key
    parents [list]: parents of node_i
    time [int]: time of interest
Returns:
	parent observations
'''
def get_parent_obs(current_gene, timeseries, parents, time):
    if not parents:
        parents = [current_gene]
    parent_obs = str([timeseries.get(parent)[time] for parent in parents])
    return parent_obs

''' Outputs the forward and backward probabilities of a given observation.
Arguments:
    child_gene [str]: gene_i name
	obs [tuple]: observed sequence of emitted states for gene_i, timeseries dictionary 
	probs [tuple]: initial, emission, transition probability dicts
Returns:
	F [float]: matrix of forward probabilities
	B [float]: matrix of backward probabilities
	R [float]: matrix of posterior probabilities
    likelihood_f [float]: P(obs) calculated using the forward algorithm
'''
def forward_backward(child_gene, obs, states, probs):
    # initialize
    current_obs, timeseries = obs
    trans_probs, emiss_probs, init_probs = probs
    n_states = len(states)
    T = len(current_obs)

    F = np.zeros([n_states, T])
    B = np.zeros([n_states, T])  # takes care intializing last row as log(1)

    for i in range(T):
        for q, state2 in enumerate(states):
            s2_parents = state2.parents
            for_parent_obs = get_parent_obs(child_gene, timeseries, s2_parents, i)
            
            if i == 0:
                F[q, i] = init_probs[q] + emiss_probs[q][current_obs[i]][for_parent_obs]
                
            else: 
                # iteration of F & B
                F_list, B_list = [], []
                for prev_q, state in enumerate(states):
                    s_parents = state.parents
                    back_parent_obs = get_parent_obs(child_gene, timeseries, s_parents, T-i)
                    
                    F_list.append(F[prev_q,i-1] + trans_probs[prev_q][q])
                    B_list.append(trans_probs[q][prev_q] + emiss_probs[prev_q][current_obs[T-i]][back_parent_obs] + B[prev_q, T-i])
                F_sum_term = sumLogProbsFunc(F_list)
                F[q, i] = emiss_probs[q][current_obs[i]][for_parent_obs] + F_sum_term
                B[q, T-i-1] = sumLogProbsFunc(B_list)
    
    # calculate the posterior
    numerator = F + B
    n, T = numerator.shape
    denominator = sumLogProbsFunc(np.vsplit(numerator, n))
    R = np.exp(numerator - denominator)

    likelihood_f = sumLogProbsFunc(np.hsplit(F[:,-1], n))

    return F, B, R, likelihood_f