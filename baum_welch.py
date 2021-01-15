import numpy as np

def sumLogProbs(a, b):
    " function for calculating sumLogProbs for two values/arrays (vectorized)"
    b_a = np.expand_dims(a+np.log(1+np.exp(b-a)), axis=0)
    a_b = np.expand_dims(b+np.log(1+np.exp(a-b)), axis=0)
    return np.squeeze(np.amax(np.concatenate((b_a, a_b), axis=0), axis=0))

def sumLogProbsFunc(args):
    " function for calculating sumLogProbs of list of values/arrays"
    if len(args) == 1:
        return args[0]
    else:
        sumtotal = sumLogProbs(args[0], args[1])
        for term in args[2:]:
            sumtotal = sumLogProbs(sumtotal, term)
        return sumtotal

def forward_backward(obs, configs, probs):
    # initialize
    current_obs, timeseries = obs
    trans_probs, emiss_probs, init_probs = probs
    n_configs = len(configs)
    T = len(current_obs)

    F = np.zeros([n_configs, T])
    B = np.zeros([n_configs, T])  # takes care intializing last row as log(1)

    for i in range(T):
        for cid1, config in enumerate(configs):
            conf_parents = config.parents
            for_parent_obs = str([timeseries.get(parent.gene)[i] for parent in conf_parents])
            
            if i == 0:
                F[cid1, i] = init_probs[cid1] + emiss_probs[cid1][current_obs[i]][for_parent_obs]
                
            else: 
                # iteration of F & B
                F_list, B_list = [], []
                for cid2, config2 in enumerate(configs):
                    conf2_parents = config2.parents
                    back_parent_obs = str([timeseries.get(parent.gene)[T-i] for parent in conf2_parents])
                    
                    F_list.append(F[cid2,i-1] + trans_probs[cid2][cid1])
                    B_list.append(trans_probs[cid1][cid2] + emiss_probs[cid2][current_obs[T-i]][back_parent_obs] + B[cid2,T-i])
                F_sum_term = sumLogProbsFunc(F_list)
                F[cid1, i] = emiss_probs[cid1][current_obs[i]][for_parent_obs] + F_sum_term
                B[cid1, T-i-1] = sumLogProbsFunc(B_list)
    
    # calculate the posterior
    numerator = F + B
    n, T = numerator.shape
    denominator = sumLogProbsFunc(np.vsplit(numerator, n))
    R = np.exp(numerator - denominator)

    likelihood_f = sumLogProbsFunc(np.hsplit(F[:,-1], n))

    return F, B, R, likelihood_f