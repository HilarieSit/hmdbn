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

def putative_hidden_graphs(gene, node_parents, ri):
    """ return list of graphs with all parent combinations, corresponding possible parent emissions & dict for tracking combinations """
    configs, configs_combos, chi_dicts = [], [], []
    # get powerset of parents
    for r in range(0, len(node_parents)+1):
        for config_parents in itertools.combinations(node_parents, r):
            new_network = node(gene, list(config_parents), theta_i=None)
            configs.append(new_network)

            # put combination of parent states in list corresponding to all_graphs
            all_parent_combos = [list(vals) for vals in itertools.product(ri, repeat=len(config_parents))]
            configs_combos.append(all_parent_combos)

            # chi dicts for corresponding combinations (# e.g. chi has four possibilities if one parent: 0-0, 0-1, 1-0, 1-1)
            chi_dict = {}
            for chi_index, combination in enumerate(all_parent_combos):
                chi_dict[str(list(combination))] = chi_index
            chi_dicts.append(chi_dict)

    return configs, configs_combos, chi_dicts

def initialize_prob_dicts(config_combos, Ri):
    """ initialize init/trans/emiss prob dicts corresponding to configs """
    # collection.defaultdict(dict) for initializing dict of dicts   
    init_probs = collections.defaultdict(dict)
    trans_probs = collections.defaultdict(dict)   
    emiss_probs = collections.defaultdict(lambda: collections.defaultdict(dict)) 
    initialization = 1/len(config_combos)

    for config_id, combinations in enumerate(config_combos):
        init_probs[config_id] = np.log(initialization)
        for config_id2, _ in enumerate(config_combos):
            trans_probs[config_id][config_id2] = np.log(initialization)
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
            # print(theta_matrix.shape)
            # print(bwbic_first_term.shape)
            bwbic_first_term[current_val, chi_index] += P[config_id, t] * theta_matrix[current_val, chi_index]

        bwbic_first_term = np.sum(bwbic_first_term)
        bwbic_second_term = (Gi/2)*(Ri-1)*np.log(np.sum(P[config_id,:]))
        bwbic = bwbic_first_term-bwbic_second_term
        bwbic_score.append(bwbic)


        # calculate bwbic score
        # log_theta = np.log(theta_matrix)
        # bwbic_first_term = np.sum(np.exp(theta_num)*log_theta)
        # bwbic_second_term = (Gi/2)*(Ri-1)*np.log(np.sum(P[config_id,:]))
        # bwbic_score.append(bwbic_first_term-bwbic_second_term)

    return theta_cond, emiss_probs, bwbic_score

def structural_EM(gene, timeseries, all_nodes, G2INT):
    """ return HMDBN for gene """
    current_obs = timeseries.get(gene)                  # timeseries for current gene
    T = len(current_obs)                                # length of timeseries
    ri = np.unique(current_obs)                         # possible emissions for gene
    Ri = len(ri)                                        # number of possible emissions for gene
    obs = (current_obs, timeseries)                     # group current_obs with timeseries dict (easier arg to pass)

    convergence = False
    best_bwbic_score = 0
    delta = 1e-4
    node_i = all_nodes[G2INT.get(gene)]

    reccuring_parents = []

    while not convergence:
        # 2. randomly change parents by adding or deleting parent node 
        # (introduce stats from previous - to speed up)
        parents = node_i.parents
        n_parents = len(parents)
        if bool(random.getrandbits(1)) or best_bwbic_score == 0:
            other_nodes = all_nodes.copy()
            # remove itself and parents
            other_nodes.remove(node_i)
            for parent in parents:
                other_nodes.remove(parent)
            # add random parent
            parent_gene = np.random.choice(other_nodes)          
            node_i.parents.append(parent_gene)
        else:
            # remove parents (50/50)
            if n_parents > 0:
                node_i.parents.pop(np.random.randint(0, n_parents))  

        # recurring_parents
        print([parents.gene for parents in node_i.parents])        

        # 3.1. identify putative hidden graphs (note: configs can have different combinations of parents from above)
        configs, configs_combos, chi_dicts = putative_hidden_graphs(gene, node_i.parents, ri)
        n_configs = len(configs)

        # 3.2. set initial values for P(q|x,HMDBN), A, pi / calculate theta & E
        P = (1/n_configs)*np.ones((n_configs, T))
        trans_probs, emiss_probs, init_probs = initialize_prob_dicts(configs_combos, Ri)
        _, emiss_probs, _ = calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P)
        probs = (trans_probs, emiss_probs, init_probs)
        
        F, B, P, _ = forward_backward(obs, configs, probs)

        # 3.3. iteratively re-estimate transition parameter to improve P(q)
        q_convergence = False
        prev_likelihood = 0
        while not q_convergence:
            # calculate probability of config h given x & HMDBN
            init_probs, trans_probs = update_probs(obs, configs, configs_combos, probs, F, B)
            theta_cond, emiss_probs, bwbic_score = calculate_theta(obs, configs, configs_combos, chi_dicts, emiss_probs, P)
            probs = (trans_probs, emiss_probs, init_probs)

            # forward backward algorithm
            F, B, P, likelihood = forward_backward(obs, configs, probs)
            
            if likelihood - prev_likelihood < delta:
                q_convergence = True
            prev_likelihood = likelihood

        # print([parents.gene for parents in node_i.parents])
        print(bwbic_score)
        print('[=========================] converged ')

        overall_bwbic_score = np.sum(bwbic_score)
        print(overall_bwbic_score)
        bwbic_ind = np.argmax(bwbic_score)
        node_i.parents = [parents for parents in configs[bwbic_ind].parents]

        # 3.4 Calculate the BWBIC score on converged P, theta
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
    hmdbn = structural_EM('twi', timeseries, all_nodes, G2INT)
