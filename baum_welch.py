import numpy as np

def sumLogProbs(a, b):
    return np.max([a+np.log(1+np.exp(b-a)), b+np.log(1+np.exp(a-b))])

def sumLogProbsFunc(*args):
    sumtotal = sumLogProbs(args[0], args[1])
    for term in args[2:]:
        sumtotal = sumLogProbs(sumtotal, term)
    return sumtotal

def forward_backward(obs, trans_probs, emiss_probs, init_probs):
    # initialize
    n_row = len(trans_probs)
    n_col = len(obs)

    F = np.zeros([n_row, n_col])
    B = np.zeros([n_row, n_col])  # takes care intializing last row as log(1)

    states = init_probs.keys()

    for i in range(n_col):
        for k, state in enumerate(states):
            if i == 0:
                # initialize F matrix
                F[k, i] = init_probs[state] + emiss_probs[state][obs[i]]
            else: 
                # iteration of F & B
                F_list, B_list = [], []
                for j, state2 in enumerate(states):
                    F_list.append(F[j,i-1] + trans_probs[state2][state])
                    B_list.append(trans_probs[state][state2] + emiss_probs[state2][obs[n_col-i]] + B[j,n_col-i])
                F_sum_term = sumLogProbsFunc(F_list)
                F[k, i] = emiss_probs[state][obs[i]] + F_sum_term
                B[k, n_col-i-1] = sumLogProbsFunc(B_list)
    
    # calculate the posterior
    numerator = F + B
    n, _ = numerator.shape
    denominator = sumLogProbsFunc(np.vsplit(numerator, n))
    R = np.exp(numerator - denominator)

    return F, B, R