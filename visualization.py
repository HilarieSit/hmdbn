import matplotlib.pyplot as plt
import numpy as np
from hmdbn import *

def plot_puntative_graphs(node_parents, P_list, states, P):
    fig, axs = plt.subplots(len(node_parents)+1)
    fig.suptitle('Initial Posterior Distribution')

    for i, iP in enumerate(P_list):
        parents = node_parents[i]
        c, _ = iP.shape
        for curve in range(c):
            axs[i].plot(np.linspace(1, 65, 65), iP[curve,:])
            axs[i].legend([parents, 'No parents'])

    c, _ = P.shape
    for curve in range(c):
        axs[i+1].plot(np.linspace(1, 65, 65), P[curve,:])
        axs[i+1].legend(states)
    
    plt.show()

def plot_posteriors(grn):
    plt.clf()
    n_subplots = len(grn)
    fig, axs = plt.subplots(n_subplots, figsize=(15,10))
    axs[0].set_title('Posterior distribution of HMDBNs')
    for i, (gene, model) in enumerate(grn.items()):
        P = model.posterior
        hid_states = [str(states).replace('[]', 'no parents').replace('[','').replace(']','').replace('\'','') for states in model.hid_states]
        c, _ = P.shape
        for curve in range(c): 
            axs[i].plot(np.linspace(1, 65, 65), P[curve,:])

        axs[i].axvspan(1, 30, alpha=0.05, color='red')
        axs[i].axvspan(30, 40, alpha=0.05, color='green')
        axs[i].axvspan(40, 58, alpha=0.05, color='blue')
        axs[i].axvspan(58, 65, alpha=0.05, color='purple')
        axs[i].legend(hid_states, bbox_to_anchor=(1, 1.05), loc='upper left', title="hidden states",)
        axs[i].get_xaxis().set_visible(False)
        axs[i].set_ylabel(model.gene)
        axs[i].set_xlim([1, 65])
 
    axs[i].get_xaxis().set_visible(True)
    axs[i].set_xlabel('time points')
    plt.show()

if __name__ == '__main__':
    hmdbns = load_hmdbns('models/final_hmdbns')
    plot_posteriors(hmdbns)

