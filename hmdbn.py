import os
from os import walk
import pickle

class hidden_state:
    def __init__(self, gene, parents=[]):
        self.gene = gene                     # gene id
        self.parents = parents               # parent nodes/graphs in stationary network

class hmdbn:
    def __init__(self, gene, parents=[], hid_states=None, pi=None, A=None, E=None, theta=None, bwbic=None, P=None):
        self.gene = gene                     # gene id
        self.parents = parents               # parent nodes/graphs in stationary network

        # track the parameters that define the fitted HMDBN
        self.hid_states = hid_states         # puntative hidden graphs
        self.pi = pi                         # initial probability dict      
        self.A = A                           # transition probability dict
        self.E = E                           # emission probability dict
        self.theta = theta                   # theta matrix
        self.bwbic = bwbic                   # bwbic score associated with HMDBN
        self.posterior = P                   # final posterior distribution   

    def save_model(self, path):
        # save parameters
        filepath = get_filepath(path, self.gene, self.parents)

        if not os.path.exists(filepath):
            os.makedirs(filepath)     

        with open(filepath+'model.pkl', 'wb') as output:
            pickle.dump([self.gene, self.parents, self.hid_states, self.theta, self.bwbic, self.posterior, self.pi, self.A, self.E], output)

def get_filepath(path, gene, parents):
    filepath = path+gene+'/'
    for i, parent in enumerate(parents):
        filepath += parent
        if i < len(parents)-1:
            filepath += '_'
    filepath += '/'
    return filepath

def load_model(path, state):
    filepath = get_filepath(path, state.gene, state.parents)
    with open(filepath+'model.pkl', "rb") as fp:
        gene, parents, hid_states, theta, bwbic, posterior, pi, A, E = pickle.load(fp)
    
    loaded_hmdbn = hmdbn(gene, parents, hid_states, pi, A, E, theta, bwbic, posterior)
    return loaded_hmdbn

def save_grn(grn, filepath):
    for hmdbn in grn.values():
        hmdbn.save_model(filepath)

def load_grn(directory):
    grn = {}
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('pkl'):
                filepath = os.path.join(dirpath, f)

                with open(filepath, "rb") as fp:
                    gene, parents, hid_states, theta, bwbic, posterior, pi, A, E = pickle.load(fp)
        
                loaded_hmdbn = hmdbn(gene, parents, hid_states, pi, A, E, theta, bwbic, posterior)
                grn[gene] = loaded_hmdbn

    return grn
        
def save_grn2(hmdbns, filepath):
    for hmdbn_i in hmdbns:
        hmdbn_i.save_model(filepath)

def load_grn(directory):
    grn = {}
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if f.endswith('pkl'):
                filepath = os.path.join(dirpath, f)

                with open(filepath, "rb") as fp:
                    gene, parents, hid_states, theta, bwbic, posterior, pi, A, E = pickle.load(fp)
        
                loaded_hmdbn = hmdbn(gene, parents, hid_states, pi, A, E, theta, bwbic, posterior)
                grn[gene] = loaded_hmdbn
    return grn
        